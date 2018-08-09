package com.skplanet.nlp.driver;

import com.skplanet.nlp.cli.CommandLineInterface;
import com.skplanet.nlp.util.FeatureFactory;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Polarity Detection Model Trainer
 *
 * - mostly based on (Multinomial) Logistc Regression in Mahout Implementation
 *
 * @author Donghun Shin / donghun.shin@sk.com
 * @since 6/24/16
 */
public class PolarityDetectionTrainer {
    private static final Logger LOGGER = Logger.getLogger(PolarityDetectionTrainer.class.getName());
    /** static variables */
    private static int word2vecDimension;
    public static FeatureFactory featureUtil;
    static Map<String, double[]> word2vec;

    public static void main(String[] args) throws IOException {
        // -------- APPLICATION ARGS -------- //
        CommandLineInterface cli = new CommandLineInterface();
        cli.addOption("t", "input", true, "training input path", true);
        cli.addOption("w", "wv", true, "word2vec path", true);
        cli.addOption("m", "model", true, "model output path", true);
        cli.addOption("d", "dimension", true, "feature size", true);
        cli.parseOptions(args);

        String trainInputPath = cli.getOption("t");
        String modelOutputPath = cli.getOption("m");
        String word2vecModelPath = cli.getOption("w");
        word2vecDimension = Integer.parseInt(cli.getOption("d"));

        // -------- Spark Context -------- //
        SparkConf conf = new SparkConf();
        conf.setAppName("Polarity Detection Trainer");
        conf.set("spark.executor.heartbeatInterval", "360");
        conf.set("spark.storage.memoryFraction", "1");
        // for local only
        conf.setMaster("local[*]");
        JavaSparkContext jsc = new JavaSparkContext(conf);


        // -------- Word2Vector Loading -------- //
        LOGGER.info("Loading Word2Vector Model ....");
        JavaPairRDD<String, double[]> word2vecPairRDD = jsc.textFile(word2vecModelPath).mapToPair(
                new PairFunction<String, String, double[]>() {
                    public Tuple2<String, double[]> call(String line) throws Exception {
                        String [] fields = line.replace("[", "").replace(" ]", "").split(",");
                        if (line.contains("emo") || fields.length != 2) {
                            return new Tuple2<String, double[]>("NULL", new double[word2vecDimension]);
                        }
                        double[] vectors = new double[fields[1].split(" ").length];
                        int index = 0;
                        for (String elem : fields[1].split(" ")) {
                            vectors[index++] = Double.parseDouble(elem);
                        }
                        return new Tuple2<String, double[]>(fields[0], vectors);
                    }
                }
        );
        word2vec = word2vecPairRDD.collectAsMap();

        //word2vec = Word2VecModel.load(jsc.sc(), word2vecModelPath);

        LOGGER.info("Loading Word2Vector Model done");


        featureUtil = new FeatureFactory(word2vec, word2vecDimension);
        if (featureUtil == null) {
            LOGGER.info("feature Utility is null");
        }

        // -------- Load Training Raw File and Generate list of LabeledPoint -------- //
        JavaRDD<LabeledPoint> trainingRDD = jsc.textFile(trainInputPath).flatMap(
                new FlatMapFunction<String, LabeledPoint>() {
                    public Iterable<LabeledPoint> call(String line) {

                        String[] fields = line.split("\\t");
                        //0> 251756181
                        //1> 1
                        //2> NULL
                        //3> 좋다
                        //4> 1
                        //5> 같/va 은/etm 제품/nng 사/vv ㅆ다가/ec
                        List<LabeledPoint> trainEntries = new ArrayList<LabeledPoint>();

                        String contno = fields[0];
                        String sentno = fields[1];
                        LOGGER.info("processing training example: " + contno + "_" + sentno);

                        List<String> tokenList = new ArrayList<String>();
                        for (String token : fields[2].split(" ")) {
                            tokenList.add(token);
                        }

                        // generate feature vector for this sentence
                        for (int i = 0; i < tokenList.size(); i++) {
                            if (tokenList.get(i).contains("_") && !tokenList.get(i).contains("/s")) {
                                double[] features = featureUtil.getContextFeature(tokenList, i);
                                try {
                                    trainEntries.add(new LabeledPoint(valueConverter(Integer.parseInt(tokenList.get(i).split("_")[1])), Vectors.dense(features)));
                                } catch (NumberFormatException e) {
                                    continue;
                                }
                            }
                        }

                        return trainEntries;
                    }
                }
        );

        // Split initial RDD into two training and testing
        JavaRDD<LabeledPoint>[] splits = trainingRDD.randomSplit(new double[]{0.7, 0.3}, 11L);

        // training data
        JavaRDD<LabeledPoint> training = splits[0].map(
                new Function<LabeledPoint, LabeledPoint>() {
                    public LabeledPoint call(LabeledPoint v1) throws Exception {
                        if (v1.features().size() == word2vecDimension) {
                            return v1;
                        } else {
                            return null;
                        }
                    }
                }
        );
        JavaRDD<LabeledPoint> trainingValid = training.cache();

        // testing data
        JavaRDD<LabeledPoint> testing = splits[1].map(
                new Function<LabeledPoint, LabeledPoint>() {
                    public LabeledPoint call(LabeledPoint v1) throws Exception {
                        if (v1.features().size() == word2vecDimension) {
                            return v1;
                        } else {
                            return null;
                        }
                    }
                }
        );
        JavaRDD<LabeledPoint> testingValid = testing.cache();

        // Building LR Model.
        final LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
                .setNumClasses(5)
                .run(trainingValid.rdd());

        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = testingValid.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Double prediction = model.predict(p.features());
                        return new Tuple2<Object, Object>(prediction, p.label());
                    }
                }
        );

        // Get evaluation metrics.
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        double precision = metrics.precision();
        LOGGER.info("Precision = " + precision);

        // Save and load model
        model.save(jsc.sc(), modelOutputPath);
    }

    /**
     * Regression Input Validation requires value to be both positive and consecutive
     * @param val org. value
     * @return converted polarity value
     */
    static int valueConverter(int val) {
        if (val == -2) {
            return 0;
        } else if (val == -1) {
            return 1;
        } else if (val == 0) {
            return 2;
        } else if (val == 1) {
            return 3;
        } else { //(val == 2)
            return 4;
        }
    }


}

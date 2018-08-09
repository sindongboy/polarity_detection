package com.skplanet.nlp.driver;

import com.skplanet.nlp.NLPAPI;
import com.skplanet.nlp.NLPDoc;
import com.skplanet.nlp.cli.CommandLineInterface;
import com.skplanet.nlp.config.Configuration;
import com.skplanet.nlp.morph.Morphs;
import com.skplanet.nlp.util.FeatureFactory;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.*;
import java.util.*;

/**
 * @author Donghun Shin / donghun.shin@sk.com
 * @since 6/25/16
 */
public class PolarityDetectionTester {
    private static final Logger LOGGER = Logger.getLogger(PolarityDetectionTester.class.getName());

    private static Map<String, double[]> word2vecModel = new HashMap<String, double[]>();

    private static int word2vecDimension = 0;

    private final static NLPAPI nlp = new NLPAPI("nlp_api.properties", Configuration.CLASSPATH_LOAD);

    public static void main(String[] args) throws IOException {
        CommandLineInterface cli = new CommandLineInterface();
        cli.addOption("i", "input", true, "test file path", true);
        cli.addOption("m", "model", true, "regression model path", true);
        cli.addOption("w", "wv", true, "word2vec model path", true);
        cli.addOption("o", "output", true, "output file path", true);
        cli.parseOptions(args);

        String modelPath = cli.getOption("m");
        String w2vModelPath = cli.getOption("w");
        File testFile = new File(cli.getOption("i"));
        File outputFile = new File(cli.getOption("o"));

        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));

        // -------- Spark Context -------- //
        SparkConf conf = new SparkConf();
        conf.setAppName("Polarity Detection Trainer");
        // for local only
        conf.setMaster("local[*]");
        JavaSparkContext jsc = new JavaSparkContext(conf);

        LOGGER.info("Loading Word2Vector Model ....");
        List<String> word2VecList = jsc.textFile(w2vModelPath).collect();
        int count = 0;
        for(String line : word2VecList) {
            if (line.trim().length() == 0) {
                continue;
            }
            if (count % 1000 == 0) {
                LOGGER.info("Word2Vec Model loading: " + count);
            }
            count++;
            line = line.replace("[", "").replace(" ]", "").replace(",", "\t");

            String[] fields = line.split("\\t");
            String[] vectorString = fields[1].split(" ");
            word2vecDimension = vectorString.length;
            double[] array = new double[vectorString.length];
            for (int i = 0; i < vectorString.length; i++) {
                array[i] = Double.parseDouble(vectorString[i]);
            }
            word2vecModel.put(fields[0], array);
        }

        //word2vecModel = Word2VecModel.load(jsc.sc(), w2vModelPath);
        LOGGER.info("Loading Word2Vector Model done");



        FeatureFactory featureUtil = new FeatureFactory(word2vecModel, word2vecDimension);

        // loading model
        LOGGER.info("Regression Model Loading ....");
        LogisticRegressionModel model = LogisticRegressionModel.load(jsc.sc(), modelPath);
        LOGGER.info("Regression Model Loading done");

        BufferedReader testReader = new BufferedReader(new FileReader(testFile));

        count = 0;
        String line;
        while ((line = testReader.readLine()) != null) {
            if (count % 1000 == 0) {
                LOGGER.info("Testing: " + count);
            }
            count++;

            String[] fields = line.split("\\t");
            String expression = fields[0];
            String answer = fields[1];
            List<String> tokenList = new ArrayList<String>();

            for (String token : fields[2].split(" ")) {
                String[] parts = token.split("/");
                if (parts.length != 2) {
                    continue;
                }

                if (parts[1].startsWith("s") && !parts[1].equals("sf")) {
                    continue;
                }

                if (parts[1].equals("unk")) {
                    continue;
                }
                tokenList.add(token);
            }
            Collections.addAll(tokenList, fields[2].split(" "));

            //  nlp processing for the expression
            NLPDoc expNLPDoc = nlp.doNLP(expression);
            Morphs expMorphs = expNLPDoc.getMorphs();
            List<String> expNLPList = new ArrayList<String>();
            for (int i = 0; i < expMorphs.getCount(); i++) {
                expNLPList.add(expMorphs.getMorph(i).getTextStr() + "/" + expMorphs.getMorph(i).getPosStr());
            }

            writer.write(expression);
            writer.write("\t");
            writer.write(answer);
            writer.write("\t");
            for (int i = 0; i < tokenList.size(); i++) {
                if (!expNLPList.contains(tokenList.get(i))) {
                    writer.write(tokenList.get(i) + " ");
                    continue;
                }

                String[] parts = tokenList.get(i).split("/");
                if (!(parts[1].equals("vv") || parts[1].equals("va"))) {
                    writer.write(tokenList.get(i) + " ");
                    continue;
                }

                double[] features = featureUtil.getContextFeature(tokenList, i);
                double result = model.predict(Vectors.dense(features));

                if (result == 0.0) {
                    result = -2;
                } else if (result == 1.0) {
                    result = -1;
                } else if (result == 2.0) {
                    result = 0;
                } else if (result == 3.0) {
                    result = 1;
                } else {
                    result = 2;
                }
                writer.write("\t" + tokenList.get(i) + "\t" + result + "\t");
            }
            writer.newLine();
        }

        testReader.close();
        writer.close();
    }
}

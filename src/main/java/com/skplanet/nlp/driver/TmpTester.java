package com.skplanet.nlp.driver;

import com.skplanet.nlp.cli.CommandLineInterface;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author Donghun Shin / donghun.shin@sk.com
 * @since 6/25/16
 */
public class TmpTester {
    private static final Logger LOGGER = Logger.getLogger(TmpTester.class.getName());

    private static Map<String, double[]> word2vecModel = new HashMap<String, double[]>();

    private static int word2vecDimension = 0;

    static boolean AVERAGE = true;


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

            /*
            String[] fields = line.split("\\t");
            String answer = fields[0];
            String content = fields[1];
            */
            String content = line;

            List<String> tokenList = new ArrayList<String>();

            for (String token : content.split(" ")) {
                tokenList.add(token);
            }



            double[] features = getContextFeature(tokenList);
            double result = model.predict(Vectors.dense(features));

            writer.write((int) result + "");
            writer.newLine();
        }

        testReader.close();
        writer.close();
    }

    static double[] getFeatures(List<String> tokenList, int index) {
        if (!getTokens(tokenList, index)) {
            return new double[word2vecDimension];
        } else {
            double[] result = word2vecModel.get(tokenList.get(index));
            if (result == null) {
                return new double[word2vecDimension];
            } else {
                return result;
            }
        }
    }

    static boolean getTokens(List<String> tokenList, int index) {
        if (index < 0 || index >= tokenList.size()) {
            return false;
        }
        return true;
    }


    static double[] getContextFeature(List<String> tokenList) {
        List<double[]> vectorList = new ArrayList<double[]>();

        for (int i = 0; i < tokenList.size(); i++) {
            vectorList.add(getFeatures(tokenList, i));
        }

        return wordVectorAddAll(vectorList);
    }

    static double[] wordVectorAddAll(List<double[]> wordVectorList) {
        double[] result = new double[word2vecDimension];
        for (double[] wordVector : wordVectorList) {
            for (int i = 0; i < wordVector.length; i++) {
                result[i] = result[i] + wordVector[i];
            }
        }

        // use average elements?
        if (AVERAGE) {
            for (int i = 0; i < word2vecDimension; i++) {
                result[i] = result[i] / wordVectorList.size();
            }
        }

        if (result == null) {
            result = new double[word2vecDimension];
        }
        return result;
    }

}

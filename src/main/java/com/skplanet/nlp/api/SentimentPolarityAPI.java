package com.skplanet.nlp.api;

import com.skplanet.nlp.config.Configuration;
import com.skplanet.nlp.core.PolarityDetector;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.LogisticRegressionModel;

import java.util.Map;

/**
 * Sentiment Polarity Detection API
 *
 * @author Donghun Shin / donghun.shin@sk.com
 * @date 6/23/16
 */
public class SentimentPolarityAPI {
    private static final Logger LOGGER = Logger.getLogger(SentimentPolarityAPI.class.getName());

    private static Map<String, double[]> word2vecModel;
    private static LogisticRegressionModel logisticModel;


    private static PolarityDetector detector = new PolarityDetector();

    /**
     * Constructor
     *
     */
    public SentimentPolarityAPI(Configuration conf) {
        /*
        // -------- Spark Context -------- //
        SparkConf sc = new SparkConf();
        sc.setAppName("Polarity Detection Trainer");
        // for local only
        sc.setMaster("local[*]");
        JavaSparkContext jsc = new JavaSparkContext(sc);

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

*/


    }

    public int getPolarity(String[] tokenList, int index) {

        return 0;
    }
}

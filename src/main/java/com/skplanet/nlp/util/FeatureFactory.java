package com.skplanet.nlp.util;

import org.apache.log4j.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author Donghun Shin / donghun.shin@sk.com
 * @since 6/29/16
 */
public final class FeatureFactory {
    private static final Logger LOGGER = Logger.getLogger(FeatureFactory.class.getName());

    private static Map<String, double[]> word2vec = null;
    private static int word2vecDimension;
    public static final boolean AVERAGE = false;
    private static List<double[]> attList = new ArrayList<double[]>();

    /**
     * Constructor
     */
    public FeatureFactory(Map<String, double[]> w2v, int w2vDimension) {
        word2vec = w2v;
        word2vecDimension = w2vDimension;
    }

    /**
     * Get Context Feature Array for the given Token list
     * @param tokenList token list
     * @param index target index
     * @return context feature array
     */
    public double[] getContextFeature(List<String> tokenList, int index) {
        List<double[]> vectorList = new ArrayList<double[]>();

        // context vector position at -3
        vectorList.add(getFeatures(tokenList, index - 3));

        // context vector position at -2
        vectorList.add(getFeatures(tokenList, index - 2));

        // context vector position at -1
        vectorList.add(getFeatures(tokenList, index - 1));

        // target vector
        vectorList.add(getFeatures(tokenList, index));

        // context vector position at +1
        vectorList.add(getFeatures(tokenList, index + 1));

        // context vector position at +2
        vectorList.add(getFeatures(tokenList, index + 2));

        // context vector position at +3
        vectorList.add(getFeatures(tokenList, index + 3));

        // backword negation clues
        double[] backNeg = findBackwordNeg(tokenList.subList(index + 1, tokenList.size()));
        vectorList.add(backNeg);

        // ji negation clues
        double[] jiNeg = findJiNeg(tokenList.subList(index + 1, tokenList.size()));
        vectorList.add(jiNeg);

        // forward negation clues
        double[] frontNeg = findforwardNeg(tokenList.subList(0, index));
        vectorList.add(frontNeg);

        // if attribute context available
        if (attList.size() > 0) {
            for (double[] attFeature : attList) {
                vectorList.add(attFeature);
            }
        }

        return wordVectorAddAll(vectorList);
    }

    /**
     * Add Attribute List if available
     * @param attributeList {@link List<String>} attribute list
     */
    public void addAttributeAll(List<String> attributeList) {
        for (String att : attributeList) {
            double[] result = word2vec.get(att);
            if (result == null) {
                attList.add(new double[word2vecDimension]);
            } else {
                attList.add(result);
            }
        }
    }

    /**
     * Add Attribute if available
     * @param attribute {@link String} attribute
     */
    public void addAttribute(String attribute) {
        /*
        try {
            attList.add(word2vec.transform(attribute).toArray());
        } catch (IllegalStateException e) {
            return;
        }
        */
        double[] result = word2vec.get(attribute);
        if (result == null) {
            attList.add(new double[word2vecDimension]);
        } else {
            attList.add(result);
        }
    }

    /**
     * Get Collective Word Vectors from context word vectors
     * @param wordVectorList word vector list
     * @return result word vectors
     */
    private double[] wordVectorAddAll(List<double[]> wordVectorList) {
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

    private double[] getFeatures(List<String> tokenList, int index) {
        if (!getTokens(tokenList, index)) {
            return new double[word2vecDimension];
        } else {
            double[] result = word2vec.get(tokenList.get(index));
            if (result == null) {
                return new double[word2vecDimension];
            } else {
                return result;
            }
        }
    }

    private boolean getTokens(List<String> tokenList, int index) {
        if (index < 0 || index >= tokenList.size()) {
            return false;
        }
        return true;
    }

    private double[] findBackwordNeg(List<String> subList) {
        // TODO: 종결어미(ef) 를 만나면 탐색을 중지해야 한다
        List<double[]> featureList = new ArrayList<double[]>();
        for (String sub : subList) {
            if (sub.equals("않/vv") || sub.equals("않/vx") || sub.equals("아니/va")) {
                double[] tmp = word2vec.get(sub);
                if (tmp != null) {
                    featureList.add(tmp);
                }
            }
        }

        if (featureList.size() == 0) {
            featureList.add(new double[word2vecDimension]);
        }

        return wordVectorAddAll(featureList);
    }

    private double[] findJiNeg(List<String> subList) {
        // TODO: 종결어미(ef) 혹은 내용어(nng/nnp)를 만나면 탐색을 중지해야 한다
        List<double[]> featureList = new ArrayList<double[]>();
        for (String sub : subList) {
            if (
                    sub.equals("지/ec") ||
                            sub.equals("지는/ec") ||
                            sub.equals("지가/ec"))
            {
                double[] tmp = word2vec.get(sub);
                if (tmp != null) {
                    featureList.add(tmp);
                }
            }
        }
        if (featureList.size() == 0) {
            featureList.add(new double[word2vecDimension]);
        }

        return wordVectorAddAll(featureList);
    }

    private double[] findforwardNeg(List<String> subList) {
        List<double[]> featureList = new ArrayList<double[]>();
        for (String sub : subList) {
            if (sub.equals("안/mag") || sub.equals("못/mag")) {
                double[] tmp = word2vec.get(sub);
                if (tmp != null) {
                    featureList.add(tmp);
                }
            }
        }
        if (featureList.size() == 0) {
            featureList.add(new double[word2vecDimension]);
        }

        return wordVectorAddAll(featureList);
    }


}

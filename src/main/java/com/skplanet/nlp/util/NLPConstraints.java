package com.skplanet.nlp.util;

/**
 * NLP Constraints
 *
 *  - filter out nlp result by its pos-tag or morph
 *
 * @author Donghun Shin / donghun.shin@sk.com
 * @since 6/23/16
 */
public final class NLPConstraints {

    public static boolean checkValidPOSTAG(String token) {
        String[] parts = token.split("/");

        if (parts.length != 2) {
            return false;
        }

        // 종결 문자를 제외한 모든 symbol 제외
        if (parts[1].startsWith("s") && !parts[1].equals("sf")) {
            return false;
        }

        // unknown words 제외
        if (parts[1].equals("unk")) {
            return false;
        }

        return true;
    }
}

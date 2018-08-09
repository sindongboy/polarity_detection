package com.skplanet.nlp.driver;

import com.skplanet.nlp.NLPAPI;
import com.skplanet.nlp.NLPDoc;
import com.skplanet.nlp.cli.CommandLineInterface;
import com.skplanet.nlp.config.Configuration;
import com.skplanet.nlp.morph.Morphs;
import org.apache.log4j.Logger;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Donghun Shin / donghun.shin@sk.com
 * @since 7/1/16
 */
public class GenerateTrainingSet {

    private static final Logger LOGGER = Logger.getLogger(GenerateTrainingSet.class.getName());

    public static void main(String[] args) throws IOException {
        CommandLineInterface cli = new CommandLineInterface();
        cli.addOption("i", "input", true, "raw input file", true);
        cli.addOption("o", "output", true, "output file", true);
        cli.parseOptions(args);

        File rawFile = new File(cli.getOption("i"));
        File outFile = new File(cli.getOption("o"));

        BufferedReader reader;
        BufferedWriter writer;

        NLPAPI nlp = new NLPAPI("nlp_api.properties", Configuration.CLASSPATH_LOAD);

        reader = new BufferedReader(new FileReader(rawFile));
        writer = new BufferedWriter(new FileWriter(outFile));

        String line;
        int count = 0;
        while ((line = reader.readLine()) != null) {
            if (line.trim().length() == 0) {
                continue;
            }

            if (count % 10000 == 0) {
                LOGGER.info("processing: " + count);
            }
            count++;

            String[] fields = line.split("\\t");
            //0> 251756181
            //1> 1
            //2> NULL
            //3> 좋다
            //4> 1
            //5> 같/va 은/etm 제품/nng 사/vv ㅆ다가/ec

            String contno = fields[0];
            String sentno = fields[1];

            String att = fields[2];
            String exp = fields[3];
            int val = Integer.parseInt(fields[4]);
            List<String> tokenList = new ArrayList<String>();
            for (String token : fields[5].split(" ")) {
                tokenList.add(token);
            }

            //  nlp processing for the attribute
            /*
            NLPDoc attNLPDoc;
            Morphs attMorphs;
            List<String> attNLPList = null;
            if (!att.equals("NULL")) {
                attNLPDoc = nlp.doNLP(att);
                attMorphs = attNLPDoc.getMorphs();
                attNLPList = new ArrayList<String>();
                for (int i = 0; i < attMorphs.getCount(); i++) {
                    attNLPList.add(attMorphs.getMorph(i).getTextStr() + "/" + attMorphs.getMorph(i).getPosStr());
                }
            }
            */

            //  nlp processing for the expression
            NLPDoc expNLPDoc = nlp.doNLP(exp);
            Morphs expMorphs = expNLPDoc.getMorphs();
            List<String> expNLPList = new ArrayList<String>();
            for (int i = 0; i < expMorphs.getCount(); i++) {
                expNLPList.add(expMorphs.getMorph(i).getTextStr() + "/" + expMorphs.getMorph(i).getPosStr());
            }

            StringBuffer sb = new StringBuffer();
            int tagCount = 0;
            for (int i = 0; i < tokenList.size(); i++) {
                if (expNLPList.contains(tokenList.get(i))) {
                    sb.append(tokenList.get(i) + "_" + val).append(" ");
                    tagCount++;
                } else {
                    sb.append(tokenList.get(i)).append(" ");
                }
            }

            if (tagCount == 0) {
                continue;
            }
            writer.write(contno + "\t");
            writer.write(sentno + "\t");
            writer.write(sb.toString().trim());
            writer.newLine();

        }
        reader.close();
        writer.close();
    }
}

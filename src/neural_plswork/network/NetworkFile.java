package neural_plswork.network;

import java.io.BufferedReader;
import java.util.LinkedList;
import java.io.IOException;
import java.util.ArrayList;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.File;


import neural_plswork.connection.initialize.PredefinedInitializer;
import neural_plswork.unit.constructor.HiddenUnitConstructor;
import neural_plswork.core.NetworkValue.NetworkValueParser;
import neural_plswork.core.NetworkValue;
import neural_plswork.math.Matrix;
import neural_plswork.math.Vector;



public class NetworkFile {
    public static boolean writeFile(Network nn, int resolution, String path, String name, boolean overwrite) {
        File file = new File(path);
        if(!file.exists()) file.mkdirs();

        int extPoint = name.indexOf(".");
        String fileName = name;
        String extension = ".nn";
        if(extPoint != -1) {
            fileName = name.substring(0, extPoint);
            extension = name.substring(extPoint);
        }

        File writeTo;

        if(!overwrite) {
            int id = 0;
            File check;
            do {
                String trueName = fileName + id + extension;
                check = new File(path + File.separator + trueName);
                id++;
            } while (check.exists());

            writeTo = check;
        } else {
            writeTo = new File(path + File.separator + fileName + extension);
        }

        return writeFile(nn, writeTo, resolution);
    }

    private static boolean writeFile(Network nn, File writeTo, int resolution) {
        try (FileWriter fw = new FileWriter(writeTo, false)) {
            String[] networkString = nn.networkToString(resolution);
            for(String s: networkString) {
                fw.write(s);
                fw.flush();
            }
            
            fw.close();

        } catch (IOException e) {
            return false;
        }

        return true;
    }

    public static boolean writeConfigFile(Network nn, String path, String name, boolean overwrite) {
        File file = new File(path);
        if(!file.exists()) file.mkdirs();

        int extPoint = name.indexOf(".");
        String fileName = name;
        String extension = ".confs";
        if(extPoint != -1) {
            fileName = name.substring(0, extPoint);
            extension = name.substring(extPoint);
        }

        File writeTo;

        if(!overwrite) {
            int id = 0;
            File check;
            do {
                String trueName = fileName + id + extension;
                check = new File(path + File.separator + trueName);
                id++;
            } while (check.exists());

            writeTo = check;
        } else {
            writeTo = new File(path + File.separator + fileName + extension);
        }

        return writeConfigFile(nn, writeTo);
    }

    private static boolean writeConfigFile(Network nn, File writeTo) {
        try (FileWriter fw = new FileWriter(writeTo, false)) {
            String[] networkString = nn.networkConfigString();
            for(String s: networkString) {
                fw.write(s);
                fw.flush();
            }
            
            fw.close();

        } catch (IOException e) {
            return false;
        }

        return true;
    }



    public static class NetworkFileBuilder extends NetworkBuilder {

        LinkedList<PredefinedInitializer[]> initializers;

        public NetworkFileBuilder(File networkFile, int MAX_THREADS, int BATCH_SIZE) {
            super(MAX_THREADS, BATCH_SIZE);
            setupInitializers(networkFile);
        }

        public NetworkFileBuilder(String networkFilePath, int MAX_THREADS, int BATCH_SIZE) {
            this(new File(networkFilePath), MAX_THREADS, BATCH_SIZE);            
        }

        private void setupInitializers(File networkFile) throws InvalidNetworkConstructionException {
            initializers = new LinkedList<>();

            try (BufferedReader br = new BufferedReader(new FileReader(networkFile))) {
                String line = br.readLine();
                
                while(line != null) {
                    int numLayers = Integer.parseInt(line);
                    initializers.addLast(constructWeights(br, numLayers));
                    line = br.readLine();
                }

            } catch (IOException e) {
                throw new InvalidNetworkConstructionException("Could not find network file: " + networkFile.getName());
            }
        }

        private PredefinedInitializer[] constructWeights(BufferedReader reader, int numLayers) throws IOException {
            PredefinedInitializer[] ret = new PredefinedInitializer[numLayers];
            
            for(int i = 0; i < numLayers; i++) {
                ret[i] = constructWeights(reader);
                // Clear "--"
                reader.readLine();
            }

            return ret;
        }

        private PredefinedInitializer constructWeights(BufferedReader reader) throws IOException {
            ArrayList<String> primaryStrings = new ArrayList<>();
            String line = reader.readLine();

            while(!line.equals("-")) {
                primaryStrings.add(line);
                line = reader.readLine();
            }

            String[] primaryArray = new String[primaryStrings.size()];
            primaryArray = primaryStrings.toArray(primaryArray);

            String biasString = reader.readLine();

            Matrix<NetworkValue> primary = NetworkValueParser.stringToMatrix(primaryArray);
            Vector<NetworkValue> bias = NetworkValueParser.stringToVector(biasString);

            return new PredefinedInitializer(primary, bias);
        }

        @Override
        public boolean appendHiddenUnit(HiddenUnitConstructor constructor, Integer[] layerSizes, Boolean[] bias, Object[]...params) throws InvalidNetworkConstructionException {
            params = prepParams(params);
            return super.appendHiddenUnit(constructor, layerSizes, bias, params);
        }

        @Override
        public boolean appendOutputUnit(Integer layerSize, Object[]...params) throws InvalidNetworkConstructionException {
            params = prepParams(params);
            return super.appendOutputUnit(layerSize, params);
        }

        private Object[][] prepParams(Object[][] params) {
            Object[][] ret = new Object[NetworkBuilder.NUM_PARAMS][];
            PredefinedInitializer[] next = null;
            if(!initializers.isEmpty()) next = initializers.pollFirst();
            if(params == null) {
                ret[INITIALIZER_INDEX] = next;
                return ret;
            }

            for(int i = 0; i < ret.length; i++) {
                if(i < params.length) ret[i] = params[i];
                else ret[i] = null;
            }
            
            if(ret[INITIALIZER_INDEX] == null) ret[INITIALIZER_INDEX] = next;

            return ret;
        }
    }
}

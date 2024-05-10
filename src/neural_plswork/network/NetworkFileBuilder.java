package neural_plswork.network;

import java.io.BufferedReader;
import java.util.LinkedList;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.io.FileReader;
import java.io.File;

import neural_plswork.neuron.activations.ActivationFunction;
import neural_plswork.connection.initialize.PredefinedInitializer;
import neural_plswork.unit.constructor.HiddenUnitConstructor;
import neural_plswork.core.NetworkValue.NetworkValueParser;
import neural_plswork.core.NetworkValue;
import neural_plswork.math.Matrix;
import neural_plswork.math.Vector;

public class NetworkFileBuilder extends NetworkBuilder {

    private static final int CLASSES = 0;
    private static final int SIZES = 1;
    private static final int BIAS = 2;
    private static final int ACTIVATIONS = 3;

    private final LinkedList<PredefinedInitializer[]> initializers;
    
    private LinkedList<Class<?>> classes;
    private LinkedList<Integer[]> sizes;
    private LinkedList<Boolean[]> bias;
    private LinkedList<ActivationFunction[]> activations;

    public NetworkFileBuilder(File networkFile, int MAX_THREADS, int BATCH_SIZE) {
        super(MAX_THREADS, BATCH_SIZE);
        this.initializers = setupInitializers(networkFile);
    }

    public NetworkFileBuilder(String networkFilePath, int MAX_THREADS, int BATCH_SIZE) {
        this(new File(networkFilePath), MAX_THREADS, BATCH_SIZE);            
    }

    public NetworkFileBuilder(File networkFile, File confsFile, int MAX_THREADS, int BATCH_SIZE) {
        this(networkFile, MAX_THREADS, BATCH_SIZE);
        uploadConfigs(confsFile);
    }

    public NetworkFileBuilder(String networkFilePath, String confsFilePath, int MAX_THREADS, int BATCH_SIZE) {
        this(new File(networkFilePath), new File(confsFilePath), MAX_THREADS, BATCH_SIZE);
    }

    @SuppressWarnings("unchecked")
    public void uploadConfigs(File configs) {
        LinkedList<?>[] confs = parseConfs(configs);
        classes = (LinkedList<Class<?>>) confs[CLASSES];
        sizes = (LinkedList<Integer[]>) confs[SIZES];
        bias = (LinkedList<Boolean[]>) confs[BIAS];
        activations = (LinkedList<ActivationFunction[]>) confs[ACTIVATIONS];
    }

    public void uploadConfigs(String configs) {
        uploadConfigs(new File(configs));
    }

    public Network buildComplete() {
        if(classes == null || sizes == null || bias == null) throw new InvalidNetworkConstructionException("Config file has not been initialized");
        System.out.println(classes.pollFirst().getName());
        defineInputLayer();
        
        try {
            while(!classes.isEmpty()) {
                Class<?> next = classes.pollFirst();
                if(next == HiddenUnitConstructor.class) {
                    HiddenUnitConstructor uc = (HiddenUnitConstructor) next.getDeclaredConstructor().newInstance();
                    appendHiddenUnit(uc);
                }
                else appendOutputUnit();
            }
        } catch (NoSuchMethodException | SecurityException | InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
            System.err.println("[ERROR] An error occurred during construction.");
            System.err.println("The build complete function only works if default constructors are available.");
            return null;
        }

        return construct();
    }

    private static LinkedList<PredefinedInitializer[]> setupInitializers(File networkFile) throws InvalidNetworkConstructionException {
        LinkedList<PredefinedInitializer[]> initializers = new LinkedList<>();

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

        return initializers;
    }

    private static PredefinedInitializer[] constructWeights(BufferedReader reader, int numLayers) throws IOException {
        PredefinedInitializer[] ret = new PredefinedInitializer[numLayers];
        
        for(int i = 0; i < numLayers; i++) {
            ret[i] = constructWeights(reader);
            // Clear "--"
            reader.readLine();
        }

        return ret;
    }

    private static PredefinedInitializer constructWeights(BufferedReader reader) throws IOException {
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

    public static LinkedList<?>[] parseConfs(File confsFile) {
        LinkedList<Class<?>> classes = new LinkedList<>();
        LinkedList<Integer[]> sizes = new LinkedList<>();
        LinkedList<Boolean[]> bias = new LinkedList<>();
        LinkedList<ActivationFunction[]> activations = new LinkedList<>();
        String line = "";
        try (BufferedReader br = new BufferedReader(new FileReader(confsFile))) {
            line = br.readLine();
            while(line != null) {
                classes.add(Class.forName(line));
                sizes.add(parseInts(br.readLine()));
                bias.add(parseBools(br.readLine()));
                activations.add(parseActivations(br.readLine()));
                line = br.readLine();
            }

        } catch (IOException e) {
            throw new InvalidNetworkConstructionException("Could not find config file: " + confsFile.getName()); 
        } catch (ClassNotFoundException e) {
            throw new InvalidNetworkConstructionException("Could not find class file for: " + line);
        }
        
        // Pop input layer activation
        if(!activations.isEmpty()) activations.pollFirst();

        return new LinkedList[] {classes, sizes, bias, activations};
    }

    private static Integer[] parseInts(String line) {
        String[] split = line.split(",");
        Integer[] ret = new Integer[split.length];

        for(int i = 0; i < split.length; i++) {
            ret[i] = Integer.parseInt(split[i]);
        }

        return ret;
    }

    private static Boolean[] parseBools(String line) {
        String[] split = line.split(",");
        Boolean[] ret = new Boolean[split.length];

        for(int i = 0; i < split.length; i++) {
            ret[i] = Boolean.parseBoolean(split[i]);
        }

        return ret;
    }

    private static ActivationFunction[] parseActivations(String line) {
        String[] split = line.split(",");
        ActivationFunction[] ret = new ActivationFunction[split.length];

        try {
            for(int i = 0; i < split.length; i++) {
                ActivationFunction af = (ActivationFunction) Class.forName(split[i]).getDeclaredConstructor().newInstance();
                ret[i] = af;
            }
        } catch (ClassNotFoundException e) {
            System.err.println("[WARNING] Could not detect one or more activation types from line: " + line);
            return null;
        } catch (NoSuchMethodException | SecurityException | InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
            System.err.println("[WARNING] Could not instantiate activation function, can only instantiate if constructor is empty");
            return null;
        }

        return ret;
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

    public boolean defineInputLayer() {
        if(classes == null || sizes == null || bias == null) throw new InvalidNetworkConstructionException("Config file has not been initialized");
        return super.defineInputLayer(sizes.pollFirst()[0], bias.pollFirst()[0]);
    }

    public boolean appendHiddenUnit(HiddenUnitConstructor constructor, Object[]... params) {
        if(classes == null || sizes == null || bias == null) throw new InvalidNetworkConstructionException("Config file has not been initialized");
        params = prepParams(params);
        return super.appendHiddenUnit(constructor, sizes.pollFirst(), bias.pollFirst(), params);
    }

    public boolean appendOutputUnit(Object[]... params) {
        if(classes == null || sizes == null || bias == null) throw new InvalidNetworkConstructionException("Config file has not been initialized");
        params = prepParams(params);
        bias.pollFirst();
        return super.appendOutputUnit(sizes.pollFirst()[0], params);
    }

    private Object[][] prepParams(Object[][] params) {
        Object[][] ret = new Object[NUM_PARAMS][];
        PredefinedInitializer[] nextInit = null;
        ActivationFunction[] nextAct = null;
        if(!initializers.isEmpty()) nextInit = initializers.pollFirst();
        if(activations != null && !activations.isEmpty()) nextAct = activations.pollFirst();
        if(params == null) {
            ret[ACTIVATION_INDEX] = nextAct;
            ret[INITIALIZER_INDEX] = nextInit;
            return ret;
        }

        for(int i = 0; i < ret.length; i++) {
            if(i < params.length) ret[i] = params[i];
            else ret[i] = null;
        }
        
        if(ret[ACTIVATION_INDEX] == null) ret[ACTIVATION_INDEX] = nextAct;
        if(ret[INITIALIZER_INDEX] == null) ret[INITIALIZER_INDEX] = nextInit;

        return ret;
    }
}

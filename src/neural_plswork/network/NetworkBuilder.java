package neural_plswork.network;

import java.util.ArrayList;
import java.util.Random;

import neural_plswork.core.NeuronLayer;
import neural_plswork.layers.basic.InputLayer;
import neural_plswork.unit.constructor.HiddenUnitConstructor;
import neural_plswork.unit.constructor.OutputUnitConstructor;
import neural_plswork.unit.ffUnits.HiddenUnit;
import neural_plswork.unit.ffUnits.OutputUnit;
import neural_plswork.neuron.activations.Activation;
import neural_plswork.neuron.activations.ActivationFunction;
import neural_plswork.neuron.activations.Sigmoid;
import neural_plswork.neuron.dropout.Dropout;
import neural_plswork.neuron.dropout.DropoutRegularizer;
import neural_plswork.neuron.dropout.NoneDropout;
import neural_plswork.neuron.evaluation.Differentiable;
import neural_plswork.neuron.evaluation.Evaluation;
import neural_plswork.neuron.evaluation.loss.Loss;
import neural_plswork.neuron.evaluation.loss.LossFunction;
import neural_plswork.connection.initialize.Initializer;
import neural_plswork.connection.initialize.UniformRandomInitializer;
import neural_plswork.connection.initialize.WeightInitializer;
import neural_plswork.connection.optimizer.OptimizationFunction;
import neural_plswork.connection.optimizer.Optimizer;
import neural_plswork.connection.penalize.Penalty;
import neural_plswork.connection.penalize.WeightPenalizer;

public class NetworkBuilder {
    private ActivationFunction DEFAULT_ACTIVATION;
    private Dropout DEFAULT_DROPOUT;
    
    private Initializer DEFAULT_INITIALIZER;
    private Penalty DEFAULT_PENALTY = Penalty.getPenalty(WeightPenalizer.NONE, 0, 0);
    private OptimizationFunction DEFAULT_OPTIMIZER = OptimizationFunction.getFunction(Optimizer.SGD);
    private final Differentiable DEFAULT_TRAINING_EVALUATOR;
    private final Evaluation DEFAULT_REPORTING_EVALUATOR;

    private Differentiable evaluator;
    private Evaluation reporter;

    private final int MAX_THREADS;
    private final int BATCH_SIZE;

    private InputLayer input;
    private ArrayList<HiddenUnit> hidden;
    private OutputUnit output;

    private Random rand;

    private double DEFAULT_L1 = 0.1;
    private double DEFAULT_L2 = 0.1;
    private double DEFAULT_DROPOUT_P = 0.1;

    public NetworkBuilder(int MAX_THREADS, int BATCH_SIZE) {
        if(MAX_THREADS < 1 || BATCH_SIZE < 1) throw new IllegalArgumentException("Must have at least 1 thread and a batch size of at least 1");
        
        this.MAX_THREADS = MAX_THREADS;
        this.BATCH_SIZE = BATCH_SIZE;

        DEFAULT_ACTIVATION = new Sigmoid();
        DEFAULT_DROPOUT = new NoneDropout();

        DEFAULT_TRAINING_EVALUATOR = (Differentiable) LossFunction.getFunction(Loss.MSE, BATCH_SIZE);
        DEFAULT_REPORTING_EVALUATOR = LossFunction.getFunction(Loss.MSE, BATCH_SIZE);

        this.evaluator = DEFAULT_TRAINING_EVALUATOR;
        this.reporter = DEFAULT_REPORTING_EVALUATOR;

        rand = new Random();
        DEFAULT_INITIALIZER = new UniformRandomInitializer(rand, -0.1, 0.1);

        hidden = new ArrayList<>();
    }

    public NetworkBuilder(int MAX_THREADS, int BATCH_SIZE, int randomSeed) {
        if(MAX_THREADS < 1 || BATCH_SIZE < 1) throw new IllegalArgumentException("Must have at least 1 thread and a batch size of at least 1");

        this.MAX_THREADS = MAX_THREADS;
        this.BATCH_SIZE = BATCH_SIZE;

        DEFAULT_TRAINING_EVALUATOR = (Differentiable) LossFunction.getFunction(Loss.MSE, BATCH_SIZE);
        DEFAULT_REPORTING_EVALUATOR = LossFunction.getFunction(Loss.MSE, BATCH_SIZE);

        this.evaluator = DEFAULT_TRAINING_EVALUATOR;
        this.reporter = DEFAULT_REPORTING_EVALUATOR;

        DEFAULT_ACTIVATION = new Sigmoid();

        rand = new Random(randomSeed);

        DEFAULT_INITIALIZER = new UniformRandomInitializer(rand, -0.1, 0.1);

        hidden = new ArrayList<>();
    }

    public boolean defineInputLayer(int layerSize, boolean bias) throws InvalidNetworkConstructionException {
        if(hidden.size() > 0) 
            throw new InvalidNetworkConstructionException("Input layer cannot be changed after Hidden Layers have begun construction");
        this.input = new InputLayer(layerSize, BATCH_SIZE, bias, MAX_THREADS);
        return true;
    }

    public boolean appendHiddenUnit(HiddenUnitConstructor constructor, Integer[] layerSizes, Boolean[] bias, Object[]...params) throws InvalidNetworkConstructionException {
        
        if(input == null) throw new InvalidNetworkConstructionException("Hidden Unit must only be created after instantiating InputLayer");
        if(output != null) throw new InvalidNetworkConstructionException("Cannot add hidden layers after output has been instantiated");
        
        Object[][] preppedParams = parseParams(params);

        ActivationFunction[] actArgs = new ActivationFunction[preppedParams[0].length];
        Dropout[] dropArgs = new Dropout[preppedParams[0].length];
        Initializer[] initArgs = new Initializer[preppedParams[1].length];
        Penalty[] penArgs = new Penalty[preppedParams[2].length];
        OptimizationFunction[] optimArgs = new OptimizationFunction[preppedParams[3].length];
        
        for(int i = 0; i < actArgs.length; i++) actArgs[i] = (ActivationFunction) preppedParams[0][i];
        for(int i = 0; i < dropArgs.length; i++) dropArgs[i] = (Dropout) preppedParams[1][i];
        for(int i = 0; i < initArgs.length; i++) initArgs[i] = (Initializer) preppedParams[2][i];
        for(int i = 0; i < penArgs.length; i++) penArgs[i] = (Penalty) preppedParams[3][i];
        for(int i = 0; i < optimArgs.length; i++) optimArgs[i] = (OptimizationFunction) preppedParams[4][i];
        
        NeuronLayer[] prior = {input};
        if(hidden.size() > 0) prior = hidden.get(hidden.size() - 1).getExitLayers();
        
        hidden.add(constructor.construct(prior, actArgs, dropArgs, layerSizes, bias, initArgs, penArgs, optimArgs, BATCH_SIZE, MAX_THREADS));
        if(!hidden.get(hidden.size() - 1).validityCheck(BATCH_SIZE, MAX_THREADS)) throw new InvalidNetworkConstructionException("Mismatching batch_size/max_threads found");
        
        return true;
    }

    public boolean appendOutputUnit(Integer layerSize, Object[]... params) throws InvalidNetworkConstructionException {

        if(input == null) throw new InvalidNetworkConstructionException("Output must only be created after instantiating InputLayer");
    
        Object[][] preppedParams = parseParams(params);

        ActivationFunction[] actArgs = new ActivationFunction[preppedParams[0].length];
        Dropout[] dropArgs = new Dropout[preppedParams[0].length];
        Initializer[] initArgs = new Initializer[preppedParams[1].length];
        Penalty[] penArgs = new Penalty[preppedParams[2].length];
        OptimizationFunction[] optimArgs = new OptimizationFunction[preppedParams[3].length];
        
        for(int i = 0; i < actArgs.length; i++) actArgs[i] = (ActivationFunction) preppedParams[0][i];
        for(int i = 0; i < dropArgs.length; i++) dropArgs[i] = (Dropout) preppedParams[1][i];
        for(int i = 0; i < initArgs.length; i++) initArgs[i] = (Initializer) preppedParams[2][i];
        for(int i = 0; i < penArgs.length; i++) penArgs[i] = (Penalty) preppedParams[3][i];
        for(int i = 0; i < optimArgs.length; i++) optimArgs[i] = (OptimizationFunction) preppedParams[4][i];
        
        NeuronLayer[] prior = {input};
        if(hidden.size() > 0) prior = hidden.get (hidden.size() - 1).getExitLayers();
        
        output = new OutputUnitConstructor(evaluator).construct(prior, actArgs, dropArgs, new Integer[]{layerSize}, new Boolean[]{}, initArgs, penArgs, optimArgs, BATCH_SIZE, MAX_THREADS);
        if(!output.validityCheck(BATCH_SIZE, MAX_THREADS)) throw new InvalidNetworkConstructionException("Mismatching batch_size/max_threads found");

        return true;
    }

    public Network construct() {
        HiddenUnit[] hiddenArray = new HiddenUnit[hidden.size()];
        return new Network(input, hidden.toArray(hiddenArray), output, evaluator, reporter, BATCH_SIZE, MAX_THREADS);
    }

    private Object[][] parseParams(Object[][] params) throws InvalidNetworkConstructionException {
        if(params == null) params = new Object[0][0];
        if(params.length > 5) throw new InvalidNetworkConstructionException("Too many arguments given");
        Object[][] ret = new Object[5][];
        for(int i = 0; i < ret.length; i++) {
            if(i < params.length && params[i] != null) {
                switch(i) {
                    case 0:
                        if(params[i] instanceof Activation[]) ret[i] = (ActivationFunction[]) convert((Activation[]) params[i]);
                        else ret[i] = params[i];
                        break;
                    case 1:
                        if(params[i] instanceof DropoutRegularizer[]) ret[i] = (Dropout[]) convert((DropoutRegularizer[]) params[i]);
                        else ret[i] = params[i];
                        break;
                    case 2:
                        if(params[i] instanceof WeightInitializer[]) ret[i] = (Initializer[]) convert((WeightInitializer[]) params[i]);
                        else ret[i] = params[i];
                        break;
                    case 3:
                        if(params[i] instanceof WeightPenalizer[]) ret[i] = (Penalty[]) convert((WeightPenalizer[]) params[i]);
                        else ret[i] = params[i];
                        break;
                    case 4:
                        if(params[i] instanceof Optimizer[]) ret[i] = convert((Optimizer[]) params[i]);
                        else ret[i] = params[i];
                        break;
                    
                }

            } else {
                Object[] column = new Object[1];
                
                switch(i) {
                    case 0:
                        column[0] = DEFAULT_ACTIVATION.copy();
                        break;
                    case 1:
                        column[0] = DEFAULT_DROPOUT.copy();
                        break;
                    case 2: 
                        column[0] = DEFAULT_INITIALIZER.copy();
                        break;
                    case 3:
                        column[0] = DEFAULT_PENALTY.copy();
                        break;
                    case 4:
                        column[0] = DEFAULT_OPTIMIZER.copy();
                        break;
                }
                

                ret[i] = column;
            }
        }

        return ret;

    }

    private ActivationFunction[] convert(Activation[] enums) {
        ActivationFunction[] ret = new ActivationFunction[enums.length];
        for(int i = 0; i < enums.length; i++) {
            ret[i] = ActivationFunction.getFunction(enums[i]);
        }

        return ret;
    }

    private Dropout[] convert(DropoutRegularizer[] enums) {
        Dropout[] ret = new Dropout[enums.length];
        for(int i = 0; i < enums.length; i++) {
            ret[i] = Dropout.getDropout(enums[i], DEFAULT_DROPOUT_P, BATCH_SIZE);
        }

        return ret;
    }

    private Initializer[] convert(WeightInitializer[] enums) {
        Initializer[] ret = new Initializer[enums.length];
        for(int i = 0; i < enums.length; i++) {
            ret[i] = Initializer.getInitializer(enums[i]);
        }

        return ret;
    }

    private Penalty[] convert(WeightPenalizer[] enums) {
        Penalty[] ret = new Penalty[enums.length];
        for(int i = 0; i < enums.length; i++) {
            ret[i] = Penalty.getPenalty(enums[i], DEFAULT_L1, DEFAULT_L2);
        }

        return ret;
    }

    private OptimizationFunction[] convert(Optimizer[] enums) {
        OptimizationFunction[] ret = new OptimizationFunction[enums.length];
        for(int i = 0; i < enums.length; i++) {
            ret[i] = OptimizationFunction.getFunction(enums[i]);
        }

        return ret;
    }

    public void setEvaluator(Loss loss) throws InvalidNetworkConstructionException {
        try {
            this.evaluator = (Differentiable) LossFunction.getFunction(loss, BATCH_SIZE);
        } catch (ClassCastException e) {
            throw new InvalidNetworkConstructionException("Evaluation function must be Differentiable");
        }
    }

    public void setEvaluator(Differentiable eval) {
        this.evaluator = eval;
    }

    public void setReporter(Loss loss) {
        this.reporter = LossFunction.getFunction(loss, BATCH_SIZE);
    }

    public void setReporter(Evaluation eval) {
        this.reporter = eval;
    }

    public void setDefaultActivation(Activation activation) {
        this.DEFAULT_ACTIVATION = ActivationFunction.getFunction(activation);
    }

    public void setDefaultActivation(ActivationFunction activation) {
        this.DEFAULT_ACTIVATION = activation;
    }

    public void setDropoutProbability(double p) {
        this.DEFAULT_DROPOUT_P = p;
    }

    public void setDefaultDropout(DropoutRegularizer dropout) {
        this.DEFAULT_DROPOUT = Dropout.getDropout(dropout, DEFAULT_DROPOUT_P, BATCH_SIZE);
    }

    public void setDefaultDropout(Dropout dropout) {
        this.DEFAULT_DROPOUT = dropout;
    }

    public void setDefaultInitializer(Initializer initializer) {
        this.DEFAULT_INITIALIZER = initializer;
    }

    public void setDefaultInitializer(WeightInitializer initializer) {
        this.DEFAULT_INITIALIZER = Initializer.getInitializer(initializer);
    }

    public void setDefaultOptimizer(OptimizationFunction optimizer) {
        this.DEFAULT_OPTIMIZER = optimizer;
    }

    public void setDefaultOptimizer(Optimizer optimizer) {
        this.DEFAULT_OPTIMIZER = OptimizationFunction.getFunction(optimizer);
    }

    public void setDefaultLambdas(double l1, double l2) {
        DEFAULT_L1 = l1;
        DEFAULT_L2 = l2;
    }

    public void setDefaultPenalty(Penalty penalty) {
        this.DEFAULT_PENALTY = penalty;
    }

    public void setDefaultPenalty(WeightPenalizer penalty) {
        this.DEFAULT_PENALTY = Penalty.getPenalty(penalty, DEFAULT_L1, DEFAULT_L2);
    }
}

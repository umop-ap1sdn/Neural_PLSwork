package neural_plswork.network;

import java.util.ArrayList;
import java.util.Random;

import neural_plswork.connection.initialize.Initializer;
import neural_plswork.connection.initialize.UniformRandomInitializer;
import neural_plswork.connection.initialize.WeightInitializer;
import neural_plswork.connection.optimizer.OptimizationFunction;
import neural_plswork.connection.optimizer.Optimizer;
import neural_plswork.connection.penalize.Penalty;
import neural_plswork.connection.penalize.WeightPenalizer;
import neural_plswork.core.NeuronLayer;
import neural_plswork.layers.basic.InputLayer;
import neural_plswork.neuron.activations.Activation;
import neural_plswork.neuron.activations.ActivationFunction;
import neural_plswork.neuron.evaluation.Differentiable;
import neural_plswork.neuron.evaluation.Evaluation;
import neural_plswork.neuron.evaluation.loss.Loss;
import neural_plswork.neuron.evaluation.loss.LossFunction;
import neural_plswork.unit.Unit;
import neural_plswork.unit.constructor.HiddenUnitConstructor;
import neural_plswork.unit.constructor.OutputUnitConstructor;
import neural_plswork.unit.ffUnits.OutputUnit;

public class NetworkBuilder {
    private final Initializer DEFAULT_INITIALIZER;
    private static final Penalty DEFAULT_PENALTY = Penalty.getPenalty(WeightPenalizer.NONE, 0, 0);
    private static final OptimizationFunction DEFAULT_OPTIMIZER = OptimizationFunction.getFunction(Optimizer.SGD);
    private final Differentiable DEFAULT_TRAINING_EVALUATOR;
    private final Evaluation DEFAULT_REPORTING_EVALUATOR;

    private Penalty penalty;
    private Differentiable evaluator;
    private Evaluation reporter;

    private final int MAX_THREADS;
    private final int BATCH_SIZE;

    private InputLayer input;
    private ArrayList<Unit> hidden;
    private OutputUnit output;

    private Random rand;

    private double DEFAULT_L1 = 0.1;
    private double DEFAULT_L2 = 0.1;

    public NetworkBuilder(int MAX_THREADS, int BATCH_SIZE) {
        this.MAX_THREADS = MAX_THREADS;
        this.BATCH_SIZE = BATCH_SIZE;

        DEFAULT_TRAINING_EVALUATOR = (Differentiable) LossFunction.getFunction(Loss.MSE, BATCH_SIZE);
        DEFAULT_REPORTING_EVALUATOR = LossFunction.getFunction(Loss.MSE, BATCH_SIZE);

        this.penalty = DEFAULT_PENALTY;
        this.evaluator = DEFAULT_TRAINING_EVALUATOR;
        this.reporter = DEFAULT_REPORTING_EVALUATOR;

        rand = new Random();
        DEFAULT_INITIALIZER = new UniformRandomInitializer(rand, -0.1, 0.1);

        hidden = new ArrayList<>();
    }

    public NetworkBuilder(int MAX_THREADS, int BATCH_SIZE, int randomSeed) {
        this.MAX_THREADS = MAX_THREADS;
        this.BATCH_SIZE = BATCH_SIZE;

        DEFAULT_TRAINING_EVALUATOR = (Differentiable) LossFunction.getFunction(Loss.MSE, BATCH_SIZE);
        DEFAULT_REPORTING_EVALUATOR = LossFunction.getFunction(Loss.MSE, BATCH_SIZE);

        this.penalty = DEFAULT_PENALTY;
        this.evaluator = DEFAULT_TRAINING_EVALUATOR;
        this.reporter = DEFAULT_REPORTING_EVALUATOR;

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

    public boolean appendHiddenUnit(HiddenUnitConstructor constructor, ActivationFunction[] activations, Integer[] layerSizes, 
        Boolean[] bias, Object[]...params) throws InvalidNetworkConstructionException {
        
        if(input == null) throw new InvalidNetworkConstructionException("Hidden Unit must only be created after instantiating InputLayer");
        if(output != null) throw new InvalidNetworkConstructionException("Cannot add hidden layers after output has been instantiated");
        
        Object[][] prepedParams = parseParams(params);

        Initializer[] initArgs = new Initializer[prepedParams[0].length];
        Penalty[] penArgs = new Penalty[prepedParams[1].length];
        OptimizationFunction[] optimArgs = new OptimizationFunction[prepedParams[2].length];

        for(int i = 0; i < initArgs.length; i++) initArgs[i] = (Initializer) prepedParams[0][i];
        for(int i = 0; i < penArgs.length; i++) penArgs[i] = (Penalty) prepedParams[1][i];
        for(int i = 0; i < optimArgs.length; i++) optimArgs[i] = (OptimizationFunction) prepedParams[2][i];
        
        NeuronLayer[] prior = {input};
        if(hidden.size() > 0) prior = hidden.get(hidden.size() - 1).getExitLayers();
        
        hidden.add(constructor.construct(prior, activations, layerSizes, bias, initArgs, penArgs, optimArgs, BATCH_SIZE, MAX_THREADS));
        
        return true;
    }

    public boolean appendHiddenUnit(HiddenUnitConstructor constructor, Activation[] activation, Integer[] layerSizes, 
        Boolean[] bias, Object[]...params) throws InvalidNetworkConstructionException {
        
        if(input == null) throw new InvalidNetworkConstructionException("Hidden Unit must only be created after instantiating InputLayer");
        if(output != null) throw new InvalidNetworkConstructionException("Cannot add hidden layers after output has been instantiated");
        
        ActivationFunction[] activations = convert(activation);
        Object[][] prepedParams = parseParams(params);

        Initializer[] initArgs = new Initializer[prepedParams[0].length];
        Penalty[] penArgs = new Penalty[prepedParams[1].length];
        OptimizationFunction[] optimArgs = new OptimizationFunction[prepedParams[2].length];

        for(int i = 0; i < initArgs.length; i++) initArgs[i] = (Initializer) prepedParams[0][i];
        for(int i = 0; i < penArgs.length; i++) penArgs[i] = (Penalty) prepedParams[1][i];
        for(int i = 0; i < optimArgs.length; i++) optimArgs[i] = (OptimizationFunction) prepedParams[2][i];
        
        
        NeuronLayer[] prior = {input};
        if(hidden.size() > 0) prior = hidden.get(hidden.size() - 1).getExitLayers();
        
        hidden.add(constructor.construct(prior, activations, layerSizes, bias, initArgs, penArgs, optimArgs, BATCH_SIZE, MAX_THREADS));
        
        return true;
    }

    public boolean appendOutputUnit(ActivationFunction activation, Integer layerSize, 
        Object[]... params) throws InvalidNetworkConstructionException {

        if(input == null) throw new InvalidNetworkConstructionException("Output must only be created after instantiating InputLayer");
    
        Object[][] prepedParams = parseParams(params);

        Initializer[] initArgs = new Initializer[prepedParams[0].length];
        Penalty[] penArgs = new Penalty[prepedParams[1].length];
        OptimizationFunction[] optimArgs = new OptimizationFunction[prepedParams[2].length];

        for(int i = 0; i < initArgs.length; i++) initArgs[i] = (Initializer) prepedParams[0][i];
        for(int i = 0; i < penArgs.length; i++) penArgs[i] = (Penalty) prepedParams[1][i];
        for(int i = 0; i < optimArgs.length; i++) optimArgs[i] = (OptimizationFunction) prepedParams[2][i];
        
        NeuronLayer[] prior = {input};
        if(hidden.size() > 0) prior = hidden.get (hidden.size() - 1).getExitLayers();
        
        output = new OutputUnitConstructor(evaluator).construct(prior, new ActivationFunction[]{activation}, new Integer[]{layerSize}, new Boolean[]{}, initArgs, penArgs, optimArgs, BATCH_SIZE, MAX_THREADS);
        
        return true;
    }

    public boolean appendOutputUnit(Activation activate, Integer layerSize, 
        Object[]... params) throws InvalidNetworkConstructionException {

        if(input == null) throw new InvalidNetworkConstructionException("Output must only be created after instantiating InputLayer");
        
        ActivationFunction[] activation = new ActivationFunction[]{ActivationFunction.getFunction(activate)};
        Object[][] prepedParams = parseParams(params);

        Initializer[] initArgs = new Initializer[prepedParams[0].length];
        Penalty[] penArgs = new Penalty[prepedParams[1].length];
        OptimizationFunction[] optimArgs = new OptimizationFunction[prepedParams[2].length];

        for(int i = 0; i < initArgs.length; i++) initArgs[i] = (Initializer) prepedParams[0][i];
        for(int i = 0; i < penArgs.length; i++) penArgs[i] = (Penalty) prepedParams[1][i];
        for(int i = 0; i < optimArgs.length; i++) optimArgs[i] = (OptimizationFunction) prepedParams[2][i];
        
        NeuronLayer[] prior = {input};
        if(hidden.size() > 0) prior = hidden.get(hidden.size() - 1).getExitLayers();
        
        output = new OutputUnitConstructor(evaluator).construct(prior, activation, new Integer[]{layerSize}, new Boolean[]{}, initArgs, penArgs, optimArgs, BATCH_SIZE, MAX_THREADS);
        
        return true;
    }

    private Object[][] parseParams(Object[][] params) throws InvalidNetworkConstructionException {
        if(params.length > 3) throw new InvalidNetworkConstructionException("Too many arguments given");
        Object[][] ret = new Object[3][];
        for(int i = 0; i < ret.length; i++) {
            if(i < params.length) {
                switch(i) {
                    case 0:
                        if(params[i] instanceof WeightInitializer[]) ret[i] = (Initializer[]) convert((WeightInitializer[]) params[i]);
                        else ret[i] = params[i];
                        break;
                    case 1:
                        if(params[i] instanceof WeightPenalizer[]) ret[i] = convert((WeightPenalizer[]) params[i]);
                        else ret[i] = params[i];
                        break;
                    case 2:
                        if(params[i] instanceof Optimizer[]) ret[i] = convert((Optimizer[]) params[i]);
                        else ret[i] = params[i];
                        break;
                    
                }

            } else {
                Object[] column = new Object[1];
                
                switch(i) {
                    case 0: 
                        column[0] = DEFAULT_INITIALIZER.copy();
                        break;
                    case 1:
                        column[0] = DEFAULT_PENALTY.copy();
                        break;
                    case 2:
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

    public void setPenalty(WeightPenalizer penalizer, double l1, double l2) {
        this.penalty = Penalty.getPenalty(penalizer, l1, l2);
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

    public void setDefaultLambdas(double l1, double l2) {
        DEFAULT_L1 = l1;
        DEFAULT_L2 = l2;
    }
}

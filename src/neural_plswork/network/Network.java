package neural_plswork.network;

import neural_plswork.connection.optimizer.OptimizationFunction;
import neural_plswork.connection.optimizer.Optimizer;
import neural_plswork.connection.penalize.Penalty;
import neural_plswork.connection.penalize.WeightPenalizer;
import neural_plswork.core.NetworkValue;
import neural_plswork.layers.basic.InputLayer;
import neural_plswork.math.Vector;
import neural_plswork.neuron.dropout.Dropout;
import neural_plswork.neuron.dropout.DropoutRegularizer;
import neural_plswork.neuron.evaluation.Differentiable;
import neural_plswork.neuron.evaluation.Evaluation;
import neural_plswork.neuron.evaluation.loss.Loss;
import neural_plswork.neuron.evaluation.loss.LossFunction;
import neural_plswork.neuron.evaluation.objective.Objective;
import neural_plswork.neuron.evaluation.objective.ObjectiveFunction;
import neural_plswork.unit.Unit;
import neural_plswork.unit.ffUnits.HiddenUnit;
import neural_plswork.unit.ffUnits.OutputUnit;

public class Network {
    private final InputLayer input;
    private final HiddenUnit[] hidden;
    private final OutputUnit output;
    
    private Differentiable evaluator;
    private Evaluation reporter;

    private final int batch_size;
    private final int max_threads;

    private double learning_rate = 0.01;

    protected Network(InputLayer input, HiddenUnit[] hidden, OutputUnit output, Differentiable evaluator, Evaluation reporter, int batch_size, int max_threads) {
        this.input = input;
        this.hidden = hidden;
        this.output = output;

        this.evaluator = evaluator;
        this.reporter = reporter;

        this.batch_size = batch_size;
        this.max_threads = max_threads;
    }

    public double[] test(double... inputs) {
        Vector<NetworkValue> vector = NetworkValue.arrToVector(inputs);
        input.setInputs(vector, 0);
        for(Unit u: hidden) u.forwardPass(0);
        output.forwardPass(0);

        return NetworkValue.vectorToArr(output.getOutputs(0));
    }

    public double[][] predict(double[][] inputs) {
        double[][] ret = new double[inputs.length][];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = test(inputs[i]);
        }

        return ret;
    }

    public Vector<NetworkValue> predict(Vector<NetworkValue> inputs, int thread) {
        input.setInputs(inputs, thread);
        for(Unit u: hidden) u.forwardPass(thread);
        output.forwardPass(thread);
        return output.getOutputs(thread);
    }

    public void calcEvals(Vector<NetworkValue> y_true, int thread, int time) {
        output.calcEvals(y_true, thread, time);
    }

    public void passEvals(int thread) {
        for(int i = hidden.length - 1; i >= 0; i--) {
            Unit next = output;
            if(i < hidden.length - 1) next = hidden[i + 1];
            hidden[i].calcEvals(next, thread);
        }
    }

    public void calculateGradients(int thread) {
        boolean descending = evaluator instanceof LossFunction;
        output.calculateGradients(learning_rate, descending, thread);
        for(int i = hidden.length - 1; i >= 0; i--) {
            hidden[i].calculateGradients(learning_rate, descending, thread);
        }
    }

    public void adjustWeights(int thread) {
        output.adjustWeights(thread);
        for(Unit u: hidden) u.adjustWeights(thread);
    }

    public void purgeEval(int thread) {
        input.purgeEval(thread);
        for(Unit u: hidden) u.purgeEval(thread);
        output.purgeEval(thread);
    }

    public void purgeEval(int thread, int times) {
        input.purgeEval(thread, times);
        for(Unit u: hidden) u.purgeEval(thread, times);
        output.purgeEval(thread, times);
    }

    public void clear(int thread) {
        input.clear(thread);
        for(Unit u: hidden) u.clear(thread);
        output.clear(thread);
    }

    public void clear() {
        input.clear();
        for(Unit u: hidden) u.clear();
        output.clear();
    }

    public double calculateEval(double[][] y_true, double[][] y_pred) {
        double eval = reporter.calculateEval(y_true, y_pred);
        eval += output.getPenaltySum();
        for(HiddenUnit h: hidden) eval += h.getPenaltySum();
        return eval;
    }

    public double calculateEval(Vector<NetworkValue>[] y_true, Vector<NetworkValue>[] y_pred) {
        double[][] y_true_arr = new double[y_true.length][];
        double[][] y_pred_arr = new double[y_pred.length][];

        for(int i = 0; i < y_true.length; i++) {
            y_true_arr[i] = NetworkValue.vectorToArr(y_true[i]);
            y_pred_arr[i] = NetworkValue.vectorToArr(y_pred[i]);
        }

        return calculateEval(y_true_arr, y_pred_arr);
    }

    

    public void setReporter(Evaluation eval) {
        this.reporter = eval;
    }

    public void setReporter(Objective objective) {
        this.reporter = ObjectiveFunction.getFunction(objective, batch_size);
    }
    
    public void setReporter(Loss loss) {
        this.reporter = LossFunction.getFunction(loss, batch_size);
    }

    public void setEvaluator(Differentiable eval) {
        this.evaluator = eval;
        this.output.setEvaluation(eval);
    }

    public void setEvaluator(Loss loss) {
        this.evaluator = (Differentiable) LossFunction.getFunction(loss, batch_size);
        this.output.setEvaluation(evaluator);
    }

    public void setDefaultLambdas(double l1, double l2, int layer) {
        if(layer > hidden.length) output.setDefaultLambdas(l1, l2);
        else hidden[layer].setDefaultLambdas(l1, l2);
    }

    public void setPenalty(Penalty[] penalty, int layer) {
        if(layer > hidden.length) output.setPenalty(penalty);
        else hidden[layer].setPenalty(penalty); 
    }

    public void setPenalty(WeightPenalizer[] penalty, int layer) {
        if(layer > hidden.length) output.setPenalty(penalty);
        else hidden[layer].setPenalty(penalty);
    }

    public void setOptimizer(OptimizationFunction[] optimizer, int layer) {
        if(layer > hidden.length) output.setOptimizer(optimizer);
        else hidden[layer].setOptimizer(optimizer);
    }

    public void setOptimizer(Optimizer[] optimizer, int layer) {
        if(layer > hidden.length) output.setOptimizer(optimizer);
        else hidden[layer].setOptimizer(optimizer);
    }

    public void setDropout(Dropout[] dropout, int layer) {
        hidden[layer].setDropout(dropout);
    }

    public void setDropout(DropoutRegularizer[] dropout, int layer) {
        hidden[layer].setDropout(dropout);
    }

    public void setAllDropout(Dropout[] dropout) {
        for(Unit u: hidden) u.setDropout(dropout);
    }

    public void setAllDropout(DropoutRegularizer[] dropout) {
        for(Unit u: hidden) u.setDropout(dropout);
    }
    public void removeDropout() {
        for(Unit u: hidden) u.setDropout(new DropoutRegularizer[]{DropoutRegularizer.NONE});
    }

    public int batch_size() {
        return batch_size;
    }

    public int max_threads() {
        return max_threads;
    }

    public void setLearningRate(double lr) {
        learning_rate = lr;
    }
}

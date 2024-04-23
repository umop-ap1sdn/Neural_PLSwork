package neural_plswork.layers.basic;

import neural_plswork.core.NetworkValue;
import neural_plswork.core.NeuronLayer;
import neural_plswork.math.Matrix;
import neural_plswork.math.Vector;
import neural_plswork.neuron.activations.ActivationFunction;
import neural_plswork.neuron.dropout.NoneDropout;
import neural_plswork.neuron.evaluation.Differentiable;

public class OutputLayer extends NeuronLayer {

    private Differentiable evaluation;

    public OutputLayer(Differentiable evaluation, ActivationFunction activation, int layerSize, int historySize, int MAX_THREADS) {
        super(activation, new NoneDropout(), layerSize, historySize, false, MAX_THREADS);
        this.evaluation = evaluation;
    }

    public Vector<NetworkValue> getOutput(int thread) {
        return super.getRecentValues(thread);
    }

    public Vector<NetworkValue> getOutput(int time, int thread) {
        return super.getValues(time, thread);
    }

    @Override
    public Vector<NetworkValue> calculateEval(Vector<NetworkValue> target, Matrix<NetworkValue> unused, int time_unused, int thread) {
        Vector<NetworkValue> eval = evaluation.calculateDerivative(target, getRecentValues(thread));
        Matrix<NetworkValue> evalMat = getRecentDerivatives(thread).multiply(eval);
        return evalMat.getAsVector();
    }

    public void setEvaluation(Differentiable evaluation) {
        this.evaluation = evaluation;
    }

}

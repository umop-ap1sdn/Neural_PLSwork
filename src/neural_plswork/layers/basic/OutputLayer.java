package neural_plswork.layers.basic;

import neural_plswork.activations.ActivationFunction;
import neural_plswork.core.NetworkValue;
import neural_plswork.core.NeuronLayer;
import neural_plswork.evaluation.Differentiable;
import neural_plswork.math.Matrix;
import neural_plswork.math.Vector;

public class OutputLayer extends NeuronLayer {

    private Differentiable evaluation;

    public OutputLayer(Differentiable evaluation, ActivationFunction activation, int layerSize, int historySize, int MAX_THREADS) {
        super(activation, layerSize, historySize, false, MAX_THREADS);
        this.evaluation = evaluation;
    }

    public Vector<NetworkValue> getOutput(int thread) {
        return super.getRecentValues(thread);
    }

    public Vector<NetworkValue> getOutput(int time, int thread) {
        return super.getValues(time, thread);
    }

    @Override
    public Vector<NetworkValue> calculateEval(Vector<NetworkValue> target, Matrix<NetworkValue> unused, int time, int thread) {
        Vector<NetworkValue> eval = evaluation.calculateDerivative(target, getValues(time, thread));
        Matrix<NetworkValue> evalMat = getDerivatives(time, thread).multiply(eval);
        return evalMat.getAsVector();
    }

    public void setEvaluation(Differentiable evaluation) {
        this.evaluation = evaluation;
    }

}

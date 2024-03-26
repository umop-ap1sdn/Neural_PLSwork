package neural_plswork.layers.basic;

import neural_plswork.activations.ActivationFunction;
import neural_plswork.core.NetworkValue;
import neural_plswork.core.NeuronLayer;
import neural_plswork.evaluation.Differentiable;
import neural_plswork.math.Matrix;
import neural_plswork.math.Vector;

public class OutputLayer extends NeuronLayer {

    private Differentiable evaluation;

    public OutputLayer(Differentiable evaluation, ActivationFunction activation, int layerSize, int historySize) {
        super(activation, layerSize, historySize, false);
        this.evaluation = evaluation;
    }

    public Vector<NetworkValue> getOutput() {
        return super.getRecentValues();
    }

    public Vector<NetworkValue> getOutput(int time) {
        return super.getValues(time);
    }

    @Override
    public Vector<NetworkValue> calculateEval(Vector<NetworkValue> target, Matrix<NetworkValue> unused, int time) {
        Vector<NetworkValue> eval = evaluation.calculateDerivative(target, getValues(time));
        Matrix<NetworkValue> evalMat = eval.pointwiseMultiply(getDerivatives(time));
        return evalMat.getAsVector();
    }

    public void setEvaluation(Differentiable evaluation) {
        this.evaluation = evaluation;
    }

}

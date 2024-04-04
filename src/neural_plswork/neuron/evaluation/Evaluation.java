package neural_plswork.neuron.evaluation;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public interface Evaluation {
    public Vector<NetworkValue> calculateEval(Vector<NetworkValue> target, Vector<NetworkValue> predicted);
    public double calculateEval(double[][] y_true, double[][] y_pred) throws IllegalArgumentException;
}

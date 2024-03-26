package neural_plswork.evaluation;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public interface Evaluation {
    public Vector<NetworkValue> calculateError(Vector<NetworkValue> target, Vector<NetworkValue> predicted);
}

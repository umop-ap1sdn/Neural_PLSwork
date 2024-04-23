package neural_plswork.neuron.evaluation.objective;

import neural_plswork.neuron.evaluation.Evaluation;
import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public interface ObjectiveFunction extends Evaluation {
    
    public static NetworkValue calculateOverallLoss(Vector<NetworkValue> errors) {
        double sum = 0;
        for(NetworkValue n: errors) sum += n.getValue();
        return new NetworkValue(sum);
    }

    public static ObjectiveFunction getFunction(Objective objective, int batchSize) throws InvalidObjectiveException {
        if(objective == null) throw new InvalidObjectiveException("Objective enum is null");
        
        switch(objective) {
            case CUSTOM: return null;
            case ROC_AUC: return null;
            case ACCURACY: return new Accuracy();
            case BINARY_ACCURACY: return null;
            case INVALID: throw new InvalidObjectiveException();
            default: throw new InvalidObjectiveException();
        }
    }
}

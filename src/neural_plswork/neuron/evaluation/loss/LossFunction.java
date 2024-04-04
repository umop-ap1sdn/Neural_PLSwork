package neural_plswork.neuron.evaluation.loss;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;
import neural_plswork.neuron.evaluation.Evaluation;

public interface LossFunction extends Evaluation {
    
    public static NetworkValue calculateOverallLoss(Vector<NetworkValue> errors) {
        double sum = 0;
        for(NetworkValue n: errors) sum += n.getValue();
        return new NetworkValue(sum);
    }

    public static LossFunction getFunction(Loss loss, int batchSize) throws InvalidLossException {
        if(loss == null) throw new InvalidLossException("Loss enum is null");
        
        switch(loss) {
            case CUSTOM: return null;
            case MSE: return new MeanSquaredError(batchSize);
            case BCE: return new BinaryCrossEntropy(batchSize);
            case CE: return new CrossEntropy(batchSize);
            case INVALID: throw new InvalidLossException();
            default: throw new InvalidLossException();
        }
    }
}

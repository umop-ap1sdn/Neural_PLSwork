package neural_plswork.regularization.penalize;

import neural_plswork.core.Copiable;
import neural_plswork.core.NetworkValue;
import neural_plswork.math.Matrix;

public interface Penalty extends Copiable {
    public Matrix<NetworkValue> getPenalty(Matrix<NetworkValue> weights);
    public Matrix<NetworkValue> getDerivative(Matrix<NetworkValue> weights);
    public Penalty copy();

    public static Penalty getPenalty(WeightPenalizer penalty, double l1, double l2) throws InvalidPenaltyException {
        if(penalty == null) throw new InvalidPenaltyException("Penalty enum is null");

        switch(penalty) {
            case CUSTOM: return null;
            case NONE: return new None();
            case LASSO: return new Lasso(l1);
            case RIDGE: return new Ridge(l2);
            case ELASTIC: return new Elastic(l1, l2);
            case INVALID: throw new InvalidPenaltyException();
            default: throw new InvalidPenaltyException();
        }
    }
}

package neural_plswork.neuron.evaluation.loss;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;
import neural_plswork.neuron.evaluation.Differentiable;

public class BinaryCrossEntropy implements LossFunction, Differentiable {
    
    private static final double EPSILON = 1e-7;
    private final int batchSize;

    protected BinaryCrossEntropy(int batchSize) {
        this.batchSize = batchSize;
    }

    @Override
    public Vector<NetworkValue> calculateEval(Vector<NetworkValue> target, Vector<NetworkValue> predicted) {
        Vector<NetworkValue> errors = new Vector<>(target.getLength());
        
        for(int i = 0; i < target.getLength(); i++) {
            double targ = target.getValue(i).getValue();
            double pred = predicted.getValue(i).getValue();
            double oneErr = Math.log((pred) + EPSILON) / Math.log(2);
            double zeroErr = Math.log((1 - pred) + EPSILON) / Math.log(2);
            double error = -1 * ((targ * oneErr) + ((1 - targ) * zeroErr)) / batchSize;
            errors.setValue(new NetworkValue(error), i);
        }

        return errors;
    }

    @Override
    public Vector<NetworkValue> calculateDerivative(Vector<NetworkValue> target, Vector<NetworkValue> predicted) {
        Vector<NetworkValue> derivs = new Vector<>(target.getLength());

        double divisor = batchSize * (Math.log(2) / Math.log(Math.E));
        
        for(int i = 0; i < target.getLength(); i++) {
            double targ = target.getValue(i).getValue();
            double pred = predicted.getValue(i).getValue();
            double oneDeriv = 1.0 / ((pred + EPSILON) * divisor);
            double zeroDeriv = -1.0 / (((1 - pred) + EPSILON) * divisor);
            double deriv = -1 * ((targ * oneDeriv) + ((1 - targ) * zeroDeriv));
            derivs.setValue(new NetworkValue(deriv), i);
        }

        return derivs;
    }


}

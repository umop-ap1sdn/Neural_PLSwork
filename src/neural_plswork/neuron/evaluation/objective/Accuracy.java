package neural_plswork.neuron.evaluation.objective;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public class Accuracy implements ObjectiveFunction {

    @Override
    public Vector<NetworkValue> calculateEval(Vector<NetworkValue> target, Vector<NetworkValue> predicted) {
        int maxI = getMax(predicted);

        NetworkValue[] acc = new NetworkValue[predicted.getLength()];
        for(int i = 0; i < target.getLength(); i++) {
            if(i == maxI) acc[i] = new NetworkValue(target.getValue(i).getValue() == 1.0 ? 1.0 : 0.0);
            else acc[i] = new NetworkValue(target.getValue(i).getValue() == 0.0 ? 1.0 : 0.0);
        }

        return new Vector<>(acc);
    }

    @Override
    public double calculateEval(double[][] y_true, double[][] y_pred) throws IllegalArgumentException {
        if(y_true.length != y_pred.length) throw new IllegalArgumentException("Arrays must be of the same length.");
        if(y_true.length == 0) return 0;
        if(y_true[0].length != y_pred[0].length) throw new IllegalArgumentException("Arrays must be of the same length.");
        
        double accuracy = 0.0;
        for(int i = 0; i < y_pred.length; i++) {
            int max = getMax(y_pred[i]);
            if(y_true[i][max] == 1.0) accuracy++;
        }

        return (accuracy / y_pred.length) * 100.0;
    }

    private int getMax(Vector<NetworkValue> outputs) {
        double max = outputs.getValue(0).getValue();
        int maxI = 0;

        for(int i = 1; i < outputs.getLength(); i++) {
            if(outputs.getValue(i).getValue() > max) {
                max = outputs.getValue(i).getValue();
                maxI = i;
            }
        }

        return maxI;
    }

    private int getMax(double[] outputs) {
        double max = outputs[0];
        int maxI = 0;

        for(int i = 1; i < outputs.length; i++) {
            if(outputs[i] > max) {
                max = outputs[i];
                maxI = i;
            }
        }

        return maxI;
    }
    
}

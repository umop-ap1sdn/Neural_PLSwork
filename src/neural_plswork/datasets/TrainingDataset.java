package neural_plswork.datasets;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public class TrainingDataset {
    
    private final Vector<NetworkValue>[] inputs;
    private final Vector<NetworkValue>[] labels;

    private final int length;
    private final int inputSize;
    private final int outputSize;

    @SuppressWarnings("unchecked")
    public TrainingDataset(double[][] inputs, double[][] labels) {
        if(inputs.length != labels.length) throw new IllegalArgumentException("Inputs array must match the size of the labels array");
        this.inputs = new Vector[inputs.length];
        this.labels = new Vector[labels.length];

        length = inputs.length;
        inputSize = inputs[0].length;
        outputSize = labels[0].length;

        initializeDataset(inputs, labels);
    }

    public TrainingDataset(Vector<NetworkValue>[] inputs, Vector<NetworkValue>[] labels) {
        this.inputs = inputs;
        this.labels = labels;

        length = inputs.length;
        inputSize = inputs[0].getRows();
        outputSize = labels[0].getRows();
    }

    private void initializeDataset(double[][] inputs, double[][] labels) {
        for(int i = 0; i < length; i++) {
            this.inputs[i] = NetworkValue.arrToVector(inputs[i]);
            this.labels[i] = NetworkValue.arrToVector(labels[i]);
        }
    }

    @SuppressWarnings("unchecked")
    public TrainingDataset slice(int start, int end) {
        Vector<NetworkValue>[] inputs = new Vector[end - start];
        Vector<NetworkValue>[] labels = new Vector[end - start];

        for(int i = 0; i < inputs.length; i++) {
            inputs[i] = this.inputs[i + start].copy();
            labels[i] = this.labels[i + start].copy();
        }

        return new TrainingDataset(inputs, labels);
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public int length() {
        return length;
    }
}

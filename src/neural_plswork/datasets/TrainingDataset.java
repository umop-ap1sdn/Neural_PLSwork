package neural_plswork.datasets;

import java.util.Iterator;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public class TrainingDataset implements Iterable<Vector<NetworkValue>[]> {
    
    protected final Vector<NetworkValue>[] inputs;
    protected final Vector<NetworkValue>[] labels;

    protected final int length;
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

    public Vector<NetworkValue> getInputSample(int index) {
        return inputs[index];
    }

    public Vector<NetworkValue> getLabelSample(int index) {
        return labels[index];
    }

    public Vector<NetworkValue>[] getInputs() {
        return inputs;
    }

    public Vector<NetworkValue>[] getLabels() {
        return labels;
    }

    @SuppressWarnings("unchecked")
    public Vector<NetworkValue>[] getSample(int index) {
        return new Vector[]{inputs[index], labels[index]};
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

    @Override
    public Iterator<Vector<NetworkValue>[]> iterator() {
        return new TrainingDatasetIterator(this);
    }

    class TrainingDatasetIterator implements Iterator<Vector<NetworkValue>[]> {
        private int index = 0;
        private final TrainingDataset td;
    
        private TrainingDatasetIterator(TrainingDataset td) {
            this.td = td;
            index = 0;
        }
    
    
        @Override
        public boolean hasNext() {
            return index < td.length;
        }
    
        @Override
        @SuppressWarnings("unchecked")
        public Vector<NetworkValue>[] next() {
            Vector<NetworkValue> input = td.inputs[index];
            Vector<NetworkValue> label = td.labels[index];
            index++;
    
            return new Vector[]{input, label};
        }
        
    }
}

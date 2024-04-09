package neural_plswork.datasets;

import java.util.Iterator;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public class TrainingDatasetIterator implements Iterator<Vector<NetworkValue>[]> {
    private int index = 0;
    private final TrainingDataset td;

    protected TrainingDatasetIterator(TrainingDataset td) {
        this.td = td;
        index = 0;
    }


    @Override
    public boolean hasNext() {
        return index < td.length();
    }

    @Override
    @SuppressWarnings("unchecked")
    public Vector<NetworkValue>[] next() {
        Vector<NetworkValue> input = td.inputs[index];
        Vector<NetworkValue> label = td.labels[index];

        return new Vector[]{input, label};
    }
    
}

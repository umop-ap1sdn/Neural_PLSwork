package neural_plswork.neuron.dropout;

import neural_plswork.core.Copiable;
import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public interface Dropout extends Copiable {

    public Vector<NetworkValue> dropout(Vector<NetworkValue> input);
    public void shuffle(int length);
    public Dropout copy();

    public static Dropout getDropout(DropoutRegularizer dr, double p, int batch_size) throws InvalidDropoutException {
        if(dr == null || dr == DropoutRegularizer.INVALID) throw new InvalidDropoutException();
        
        switch(dr) {
            case CUSTOM: return null;
            case NONE: return new NoneDropout();
            case SAMPLE_WISE: return new SampleDropout(p);
            case BATCH_WISE: return new BatchDropout(p, batch_size);
            default: return null;
        }
    }
}

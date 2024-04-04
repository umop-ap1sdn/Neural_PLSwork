package neural_plswork.connection.initialize;

import neural_plswork.core.Copiable;
import neural_plswork.core.NetworkValue;

public interface Initializer extends Copiable {
    public NetworkValue getNextWeight(int row, int column);
    public Initializer copy();

    public static Initializer getInitializer(WeightInitializer initializer) throws InvalidInitializationException {
        if(initializer == null) throw new InvalidInitializationException("Penalty enum is null");

        switch(initializer) {
            case CUSTOM: return null;
            case UNIF: return new UniformRandomInitializer();
            case NORMAL: return new UniformRandomInitializer();
            case CONSTANT: return new ConstantInitializer();
            case INVALID: throw new InvalidInitializationException();
            default: throw new InvalidInitializationException();
        }
    }
}

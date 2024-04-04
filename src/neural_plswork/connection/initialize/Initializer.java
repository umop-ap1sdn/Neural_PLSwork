package neural_plswork.connection.initialize;

import neural_plswork.core.Copiable;
import neural_plswork.core.NetworkValue;

public interface Initializer extends Copiable {
    public NetworkValue getNextWeight(int row, int column);
    public Initializer copy();

    public static Initializer getInitializer(WeightInitializer intiializer) throws InvalidInitializationException {
        if(intiializer == null) throw new InvalidInitializationException("Penalty enum is null");

        switch(intiializer) {
            case CUSTOM: return null;
            case UNIF: return new UniformRandomInitializer();
            case NORMAL: return new UniformRandomInitializer();
            case CONSTANT: return new ConstantInitializer();
            case INVALID: throw new InvalidInitializationException();
            default: throw new InvalidInitializationException();
        }
    }
}

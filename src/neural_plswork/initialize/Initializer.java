package neural_plswork.initialize;

import neural_plswork.core.NetworkValue;

public interface Initializer {
    public NetworkValue getNextWeight(int row, int column);

    public static Initializer getPenalty(WeightInitializer penalty) throws InvalidInitializationException {
        if(penalty == null) throw new InvalidInitializationException("Penalty enum is null");

        switch(penalty) {
            case CUSTOM: return null;
            case UNIF: return new UniformRandomInitializer();
            case NORMAL: return new UniformRandomInitializer();
            case CONSTANT: return new ConstantInitializer();
            case INVALID: throw new InvalidInitializationException();
            default: throw new InvalidInitializationException();
        }
    }
}

package neural_plswork.optimizer;

public enum Optimizer {
    CUSTOM(0),
    SGD(1),
    MOMENTUM(2),
    ADAM(3),
    INVALID(-1);


    private final int id;
    private Optimizer(int id) {
        this.id = id;
    }

    public Optimizer getById(int id) {
        switch(id) {
            case 0:
                return CUSTOM;
            case 1:
                return SGD;
            case 2:
                return MOMENTUM;
            case 3:
                return ADAM;
            default:
                return INVALID;
        }
    }

    @Override
    public String toString() {
        return String.valueOf(id);
    }
}

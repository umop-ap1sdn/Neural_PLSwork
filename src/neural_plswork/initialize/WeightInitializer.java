package neural_plswork.initialize;

public enum WeightInitializer {
    CUSTOM(0),
    UNIF(1),
    NORMAL(2),
    CONSTANT(3),
    INVALID(-1);

    private final int id;
    private WeightInitializer(int id) {
        this.id = id;
    }

    public WeightInitializer getById(int id) {
        switch(id) {
            case 0:
                return CUSTOM;
            case 1:
                return UNIF;
            case 2:
                return NORMAL;
            case 3:
                return CONSTANT;
            default:
                return INVALID;
        }
    }

    @Override
    public String toString() {
        return String.valueOf(id);
    }
}

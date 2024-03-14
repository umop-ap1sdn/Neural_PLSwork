package neural_plswork.loss;

public enum Loss {
    CUSTOM(0),
    SSE(1),
    MSE(2),
    BCE(3),
    INVALID(-1);


    final int id;
    private Loss(int id) {
        this.id = id;
    }

    public Loss getById(int id) {
        switch(id) {
            case 0:
                return CUSTOM;
            case 1:
                return SSE;
            case 2:
                return MSE;
            case 3:
                return BCE;
            default:
                return INVALID;
        }
    }

    @Override
    public String toString() {
        return String.valueOf(id);
    }
}

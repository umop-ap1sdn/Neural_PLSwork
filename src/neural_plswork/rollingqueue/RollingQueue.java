package neural_plswork.rollingqueue;

import java.util.Iterator;

public class RollingQueue<T> implements Iterable<T> {
    
    private T[] queue;
    private int head;
    private int tail;
    private int size;
    private final int capacity;

    @SuppressWarnings("unchecked")
    public RollingQueue(int capacity) {
        this.capacity = capacity;
        head = 0;
        tail = 0;
        size = 0;

        queue = (T[]) new Object[capacity];
    }

    public T get(int index) throws ArrayIndexOutOfBoundsException {
        if(index >= size) throw new ArrayIndexOutOfBoundsException();
        return queue[(head + index) % capacity];
    }

    public T getLast() throws IllegalStateException {
        if(size <= 0) throw new IllegalStateException();
        int index = tail - 1;
        if(index < 0) index += capacity;
        return queue[index];
    }

    public void set(int index, T value) throws ArrayIndexOutOfBoundsException{
        if(index >= size) throw new ArrayIndexOutOfBoundsException();
        queue[(index + head) % capacity] = value;
    }

    public T pop() throws IllegalStateException {
        if(size <= 0) throw new IllegalStateException("Cannot pop from empty queue");
        T ret = queue[head];
        head = (head + 1) % capacity;
        size--;
        return ret;
    }

    public T popTail() throws IllegalStateException {
        if(size <= 0) throw new IllegalStateException("Cannot pop from empty queue");
        tail--;
        if(tail < 0) tail += capacity;
        size--;
        return queue[tail];
    }

    public void push(T value) throws IllegalStateException {
        if(size >= capacity) throw new IllegalStateException("Cannot push to full queue");
        queue[tail] = value;
        tail = (tail + 1) % capacity;
        size++;
    }

    public void pushHead(T value) throws IllegalStateException {
        if(size >= capacity) throw new IllegalStateException("Cannot push to a full queue");
        head--;
        if(head < 0) head += capacity;
        queue[head] = value;
        size++;
    }

    public int size() {
        return size;
    }

    public void clear() {
        head = 0;
        tail = 0;
        size = 0;
    }

    @Override
    public Iterator<T> iterator() {
        return new RollingQueueIterator(this);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for(int i = 0; i < size; i++) {
            int index = (i + head) % capacity;
            sb.append(queue[index]);
            if(i < size - 1) sb.append(", ");
        }

        sb.append("]");
        return sb.toString();
    }

    class RollingQueueIterator implements Iterator<T> {

        private int index = 0;
        private RollingQueue<T> rq;
        RollingQueueIterator(RollingQueue<T> rq) {
            this.rq = rq;
        }

        @Override
        public boolean hasNext() {
            return (index < rq.size());
        }

        @Override
        public T next() {
            return rq.get(index++);
        }
        
    }

}

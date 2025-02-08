#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <queue>
#include <random>
#include <limits>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <optional>

#define DEBUG 0

using namespace std;
using namespace chrono;

uint64_t file_reads = 0;
uint64_t file_writes = 0;

/* 
 * StaticMinHeap: A fixed-size (deterministic) min-heap implementation for managing elements efficiently.
 * Provides basic heap operations like inserting an element, retrieving the smallest element, and removing it.
 */
template<typename T>
class StaticMinHeap {
public:

    void print(void) const {
#if DEBUG
        cout << "min Heap ";
        for (size_t i = 0; i < heap_size; ++i) {
            cout << heap[i].second << " ";
        }
        cout << endl;
#endif
    }
    
    /* Constructor: Initializes the min-heap with a specified maximum size. */
    StaticMinHeap(uint64_t max_heap_size) : heap_size(0) {
        this->max_heap_size = max_heap_size;
        heap = new pair<ifstream*, T>[max_heap_size];
    }

    /* Checks if the heap is empty. */
    bool empty() const { return heap_size == 0; }

    /* Adds a new element to the heap. Throws an exception if the heap is full. */
    void emplace(ifstream* file, T value) {
        if (heap_size < max_heap_size) {
            heap[heap_size] = make_pair(file, value);
            heapify_up(heap_size);
            ++heap_size;
        } else {
            throw overflow_error("Heap is full");
        }
        print();
    }

    /* Retrieves the smallest element in the heap without removing it. */
    pair<ifstream*, T> top() const {
        if (empty()) throw underflow_error("Heap is empty");
        print();
        return heap[0];
    }

    /* Removes the smallest element from the heap. */
    void pop() {
        if (empty()) throw underflow_error("Heap is empty");
#if DEBUG
        cout << heap[0].second << " is popped " << endl;
#endif
        heap[0] = heap[heap_size - 1];
        --heap_size;
        heapify_down(0);
    }

    /* Destructor: Frees allocated memory for the heap. */
    ~StaticMinHeap() {
        delete[] heap;
    }

private:
    uint64_t max_heap_size;
    pair<ifstream*, T> *heap;
    size_t heap_size;  /* Tracks the current number of elements in the heap */

    /* Maintains the heap property by moving an element up. */
    void heapify_up(size_t index) {
        while (index > 0 && heap[parent(index)].second > heap[index].second) {
            swap(heap[index], heap[parent(index)]);
            index = parent(index);
        }
    }

    /* Maintains the heap property by moving an element down. */
    void heapify_down(size_t index) {
        size_t left = left_child(index);
        size_t right = right_child(index);
        size_t smallest = index;

        if (left < heap_size && heap[left].second < heap[smallest].second) {
            smallest = left;
        }
        if (right < heap_size && heap[right].second < heap[smallest].second) {
            smallest = right;
        }
        if (smallest != index) {
            swap(heap[index], heap[smallest]);
            heapify_down(smallest);
        }
    }

    /* Calculates the parent index of a given index. */
    size_t parent(size_t index) {
        return (index - 1) / 2; 
    }

    /* Calculates the left child index of a given index. */
    size_t left_child(size_t index) { 
        return 2 * index + 1; 
    }

    /* Calculates the right child index of a given index. */
    size_t right_child(size_t index) { 
        return 2 * index + 2; 
    }
};

template<typename T>
class StaticMinHeapMultiLayer {
public:
    // Constructor
    StaticMinHeapMultiLayer(const vector<ifstream*>& chunk_streams, uint64_t depth) 
        : heap_size(0), depth(depth), min_heap(chunk_streams.size()) {

        umap.reserve(chunk_streams.size());

        for (ifstream* file : chunk_streams) {
            vector<T> buffer(depth, numeric_limits<T>::max());  // Pre-fill with max values
            start_indices[file] = 0;
            sizes[file] = 0;

            // Read data directly into the vector
            file_reads++;
            file->read(reinterpret_cast<char*>(buffer.data()), sizeof(T) * depth);
            streamsize bytes_read = file->gcount();  // Get the actual number of bytes read

            // Convert bytes to number of elements
            uint64_t elements_read = bytes_read / sizeof(T);
            sizes[file] = elements_read;  // Track valid elements in the buffer

            umap[file] = move(buffer);
            heap_size += elements_read;  // Update total heap size

            // Add the smallest element from this file to the min-heap
            if (elements_read > 0) {
                min_heap.emplace(file, umap[file][start_indices[file]]);
            }
        }

        print();
    }

    // Check if the heap is empty
    bool empty() const {
        return heap_size == 0;
    }

    // Get the smallest element and remove it (combined with top)
    pair<uint64_t, T> pop() {
        if (empty()) throw underflow_error("Heap is empty");

        // Get the smallest element from the min-heap
        auto [min_file, min_value] = min_heap.top();
        min_heap.pop();

        // Invalidate the popped element
        start_indices[min_file] =  (start_indices[min_file] + 1) % depth; 
        --sizes[min_file];
        --heap_size;

        if (sizes[min_file] > 0){
            min_heap.emplace(min_file, umap[min_file][start_indices[min_file]]);
        }

        print();
        return {sizes[min_file], min_value};
    }

    // Refill buffers for files that have space
    void refillFiles() {
        for (auto& [file, size] : sizes) {
            if (size < depth) {
                refillFile(file);
            }
        }
        print();
    }

    // Debugging function
    void print() {
#if DEBUG
        for (const auto& [file, vec] : umap) {
            cout << "File pointer: " << file << " -> Values: ";
            uint64_t idx = start_indices[file];
            for (size_t i = 0; i < sizes[file]; ++i) {
                cout << vec[idx] << " ";
                idx = (idx + 1) % depth;
            }
            cout << '\n';
        }
#endif
    }

private:
    uint64_t heap_size;
    uint64_t depth;
    unordered_map<ifstream*, vector<T>> umap;
    unordered_map<ifstream*, uint64_t> start_indices;  // Track start of ring buffer
    unordered_map<ifstream*, uint64_t> sizes;  // Track current size of each vector
    StaticMinHeap<T> min_heap;  // Min-heap to track the smallest elements

    // Refill the buffer for a specific file
    void refillFile(ifstream* file) {
        if (file->eof()) {
            return;
        }

        streamsize bytes_read = 0;

        vector<T>& buffer = umap[file];

        uint64_t remaining_size = depth - sizes[file];

        if (remaining_size > start_indices[file]){
            file_reads++;
            file->read(reinterpret_cast<char*>(buffer.data() + start_indices[file] + sizes[file]), sizeof(T) * (remaining_size - start_indices[file]));
            bytes_read += file->gcount();
            file_reads++;
            file->read(reinterpret_cast<char*>(buffer.data()), sizeof(T) * (start_indices[file]));
            bytes_read += file->gcount();
        } else {
            file_reads++;
            file->read(reinterpret_cast<char*>(buffer.data() + start_indices[file] + sizes[file]), sizeof(T) * (remaining_size - start_indices[file]));
            bytes_read += file->gcount();
        }

        if (bytes_read > 0) {
            uint64_t elements_read = bytes_read / sizeof(T);

            if (sizes[file] == 0 && elements_read){
                min_heap.emplace(file, umap[file][start_indices[file]]);
            }

            sizes[file] += elements_read;  

            heap_size += elements_read;
        }
    }
};

/*
 * Merges multiple sorted chunk files into a single sorted output file using a min-heap.
 */
template <typename T>
void merge_files_using_min_heap(const string& output_file, vector<string> temp_files, size_t chunk_size) {
    ofstream output(output_file, ios::binary);
    vector<ifstream*> chunk_streams;

    /* Open all temporary chunk files and push their first elements into the heap */
    for (const auto& temp_file : temp_files) {
        auto* file = new ifstream(temp_file, ios::binary);
        chunk_streams.push_back(file);
    }

    uint64_t chunk_window =  chunk_size / chunk_streams.size();

    if (chunk_window > 1){
        StaticMinHeapMultiLayer<T> min_heap_multi_layer(chunk_streams, chunk_window);

        vector<T> output_vector;
        output_vector.reserve(chunk_window);

        // Merge chunks by repeatedly extracting the smallest element from the heap
        while (!min_heap_multi_layer.empty()) {
            // Pop the smallest element from the multi-layer heap
            auto [size, value] = min_heap_multi_layer.pop();

            // Add the popped value to the output buffer
            output_vector.push_back(value);

            // Check if the output buffer is full or if the size returned by pop() is 0
            if (output_vector.size() == chunk_window || size == 0) {
                // Write the output buffer to the file
                output.write(reinterpret_cast<const char*>(output_vector.data()), output_vector.size() * sizeof(T));
                file_writes++;
                output_vector.clear();  // Clear the buffer after writing
                min_heap_multi_layer.refillFiles();
            }
        }

        // Write any remaining elements in the output buffer to the file
        if (!output_vector.empty()) {
            output.write(reinterpret_cast<const char*>(output_vector.data()), output_vector.size() * sizeof(T));
        }
    } else {
        StaticMinHeap<T> min_heap(temp_files.size());

        /* Open all temporary chunk files and push their first elements into the heap */
        for (const auto& temp_file : temp_files) {
            auto* file = new ifstream(temp_file, ios::binary);
            chunk_streams.push_back(file);

            T value;
            file->read(reinterpret_cast<char*>(&value), sizeof(T));
            size_t read_count = file->gcount() / sizeof(T);
            if (read_count) {
                min_heap.emplace(file, value);
            }
        }

        /* Merge chunks by repeatedly extracting the smallest element from the heap */
        while (!min_heap.empty()) {
            auto [file, value] = min_heap.top();
            min_heap.pop();

            /* Write the extracted value */
            output.write(reinterpret_cast<const char*>(&value), sizeof(T));   
            
            /* Read the next value from the file and push it to the min-heap */
            T next_value; 
            file->read(reinterpret_cast<char*>(&next_value), sizeof(T));
            size_t read_count = file->gcount() / sizeof(T);
            if (read_count) {
                min_heap.emplace(file, next_value);
            }
        }
    }

    /* Cleanup temporary files and resources */
    output.close();
    for (auto* file : chunk_streams) {
        file->close();
        delete file;
    }
    for (const auto& temp_file : temp_files) {
        remove(temp_file.c_str());
    }

    cout << "file reads = " << file_reads << " file writes = " << file_writes << endl;
}

/* 
 * Splits a large binary file into smaller, sorted chunks stored on disk.
 * Returns the file names of the temporary sorted chunks.
 */
template <typename T>
vector<string> wipe_out_chunks_to_the_disk(const string& input_file, size_t chunk_size) {
    /* Input validation */
    ifstream input(input_file, ios::binary);
    if (!input) {
        throw runtime_error("Error opening input file: " + input_file);
    }

    vector<string> temp_files; /* Stores the names of temporary sorted chunk files */
    size_t chunk_index = 0;    /* Tracks the index of the current chunk */

    vector<T> chunk_buffer(chunk_size); /* Buffer for the data to be chunked */

    /* Wipe the chunks to the disk */
    while (!input.eof()) {
        input.read(reinterpret_cast<char*>(chunk_buffer.data()), chunk_size * sizeof(T));
        size_t read_count = input.gcount() / sizeof(T);

        if (read_count) {
            chunk_buffer.resize(read_count);
            sort(chunk_buffer.begin(), chunk_buffer.end());

            string temp_file = "temp_chunk_" + to_string(chunk_index++) + ".bin";
            ofstream temp(temp_file, ios::binary);
            temp.write(reinterpret_cast<const char*>(chunk_buffer.data()), read_count * sizeof(T));
            temp.close();

            temp_files.push_back(temp_file);
        }
    }
    input.close();

    return temp_files;
}

/* 
 * Performs external merge sort on a large binary file. 
 * Splits the input file into sorted chunks and merges them into a single sorted output file.
 */
template <typename T>
void sort_large_file(const string& input_file, const string& output_file, size_t chunk_size) 
{
    /* Step 1: Split the input file into sorted chunks */
    vector<string> temp_files = wipe_out_chunks_to_the_disk<T>(input_file, chunk_size);

    /* Step 2: Merge sorted chunks using a min-heap */
    merge_files_using_min_heap<T>(output_file, temp_files, chunk_size);
}

#if DEBUG
/* 
 * Main function for argument handling and executing the sort.
 */
int main() {
    string input_file = "test_data.bin";
    string output_file = "output_data.bin";
    size_t chunk_size = 100000;
    string type = "uint64";

    if (!chunk_size) {
        throw runtime_error("Error chunk_size cannot be 0");
    }

    if (!(type == "float" || type == "uint64")) {
        throw runtime_error("Invalid type");
    }

    bool is_floating_point = (type == "float");

    /* Calculate the RAM needed for the process */
    ifstream file(input_file, ios::binary | ios::ate);
    if (!file.is_open()) {
        throw runtime_error("Unable to open file: " + input_file);
    }
    
    /* Get the size of the file */
    streamsize file_size = file.tellg();
    file.close();

    uint16_t type_size = is_floating_point ? sizeof(float) : sizeof(uint64_t);

    cout << "Approximate RAM will be in use = " 
     << chunk_size * type_size 
     + static_cast<size_t>(ceil(static_cast<double>(file_size) / (chunk_size * type_size))) * type_size 
     << " bytes\n";

    try {
        auto start_time = high_resolution_clock::now();
        if (is_floating_point){
            sort_large_file<float>(input_file, output_file, chunk_size);
        } else {
            sort_large_file<uint64_t>(input_file, output_file, chunk_size);            
        }

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);

        cout << "Sorting complete. Output written to " << output_file << "\n";
        cout << "Time taken: " << duration.count() << " milliseconds\n";
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
#else
/* 
 * Main function for argument handling and executing the sort.
 */
int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file> <chunk_size> <type>\n";
        cerr << "type: 'float' or 'uint64'\n";
        cerr << "where chunk_size ~= RAM / sizeof(type)\n";
        return 1;
    }

    string input_file = argv[1];
    string output_file = argv[2];
    size_t chunk_size = stoull(argv[3]);
    string type = argv[4];

    if (!chunk_size) {
        throw runtime_error("Error chunk_size cannot be 0");
    }

    if (!(type == "float" || type == "uint64")) {
        throw runtime_error("Invalid type");
    }

    bool is_floating_point = (type == "float");

    /* Calculate the RAM needed for the process */
    ifstream file(input_file, ios::binary | ios::ate);
    if (!file.is_open()) {
        throw runtime_error("Unable to open file: " + input_file);
    }
    
    /* Get the size of the file */
    streamsize file_size = file.tellg();
    file.close();

    uint16_t type_size = is_floating_point ? sizeof(float) : sizeof(uint64_t);

    cout << "Approximate RAM will be in use = " 
     << chunk_size * type_size 
     + static_cast<size_t>(ceil(static_cast<double>(file_size) / (chunk_size * type_size))) * type_size 
     << " bytes\n";

    try {
        auto start_time = high_resolution_clock::now();
        if (is_floating_point){
            sort_large_file<float>(input_file, output_file, chunk_size);
        } else {
            sort_large_file<uint64_t>(input_file, output_file, chunk_size);            
        }

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);

        cout << "Sorting complete. Output written to " << output_file << "\n";
        cout << "Time taken: " << duration.count() << " milliseconds\n";
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
#endif

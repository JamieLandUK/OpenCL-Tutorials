// Reduce Add - summation on all elements to calculate the mean.
kernel void reduce_add(global const int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(id % (i * 2)) && ((id + i) < N)) 
			B[id] += B[id + i];

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}


// Bitonic sort - sorts through multiple steps.
// Compare and Change

kernel void cmpxchg(global int* A, global int* B, global bool* dir) {
	// When downsweeping and the content of A is bigger than content of B...
	// or when upsweeping and the content of B is bigger than the content of A...
	if ((!dir && *A > *B) || (dir && *A < *B)) {
		// Swap them.
		int t = *A; *A = *B; *B = t;
	}
}

// Merge sort
kernel void bitonic_merge(int id, int N, global int* A, global bool* dir) {
	for (int i = N / 2; i > 0; i /= 2) {
		if ((id % (i * 2)) < i) {
			cmpxchg(&A[id], &A[id + 1], dir);

			barrier(CLK_GLOBAL_MEM_FENCE);
		}
	}
}

// The main content of the bitonic sort implementation.
kernel void bitonic_sort(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	// Setting a for loop through strides
	for (int i = 1; i < N/2; i *= 2) {
		if (id % (i * 4) < i * 2) {
			bitonic_merge(id, A, i * 2, false); // Downsweep
		}
		else if ((id + i * 2) % (i * 4) < i * 2) {
			bitonic_merge(id, A, i * 2, true); // Upsweep
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	// Final merge sort to change it from a bitonic sequence to a sorted vector
	// Before there was 'if (id==0)' however this was never running the final sort at all.
	// This might be bad for total performance though.
	bitonic_merge(id, A, N, false);
}

// Min and Max
kernel void min_max(global int* A, global int* C) {
	C[0] = A[get_global_id(0)]; // first element
	C[1] = A[get_global_id(-1)]; // last element
}
#include <iostream>
#include <vector>
#include <iomanip>

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

/* Program description:
 * This main content of this program will accept in the file defined on line 77, which is read in from the Debug folder.
 * It creates a vector which is then filled with the final portion of each line which is read in sequentially. This is faster than
 * reading each portion of each line sequentially due to a smaller amount of reads.
 * After that, it creates kernels for each function: a sort function (used to find median, 1st and 3rd quartiles),
 * a reduce function that adds all of data together to find a mean (taken from Lecture 7's sorting algorithms and then updated
 * to manage 
 */

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0;}
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		// need CL_QUEUE_PROFILING_ENABLE to receive ns time reports.
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef int mytype;

		// temporary values
		string temp; // Holds entire strings
		string portion; // Holds entire words

		std::ifstream file; // The input file.
		std::vector<mytype> A; // Our main program vector
		std::vector<string> vec; // A vector which holds each word from each line.

		// "temp_lincolnshire_shorter.txt" is 51 records long.
		// "temp_lincolnshire_short.txt" is 18732 records long.
		// "temp_lincolnshire.txt" is 1873107 records long.
		// Can be changed to point to a different file here.
		file.open("temp_lincolnshire_short.txt", ios::in);

		// If the file hasn't been opened properly
		// (or doesn't exist)...
		if (!file) {
			// Tell the user.
			std::cerr << "Could not read file!" << std::endl;
			// End the program.
			return 1;
		}

		// file.good finds if the current line of the file exists.
		// It is similar to .eof(), however that seems to have some issues.
		while (file.good()) {
			// Get the first line of our file and load it into temp.
			while (std::getline(file, temp)) {
				// Turn temp back into a proper string (string stream)
				stringstream ss(temp);
				// Which means we can create another while loop which splits
				// the string into words, splitting on whitespace.
				while (std::getline(ss, portion, ' ')) {
					// Push the word into the vector.
					vec.push_back(portion);
				}
				// We only want the last data item from the vector
				// And it needs to be changed into a float and then pushed into the new vector.
				A.push_back(stof(vec.back()));
				vec.clear(); // Clear the intermediary vector to save space.
			}
		}

		file.close(); // Close the file.
		
		
		// We need the kernels to be defined here so we can get the work group info next.
		cl::Kernel kernel_reduceadd = cl::Kernel(program, "reduce_add");
		cl::Kernel kernel_bitonicsort = cl::Kernel(program, "bitonic_sort");
		cl::Kernel kernel_minmax = cl::Kernel(program, "min_max");

		// Get the device from the context which will be used later to decide our local size.
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];

		cl::Event total_profile; // Gets the kernel execution time for all kernels
		cl::Event A_profile; 
		cl::Event B_profile; 
		cl::Event C_profile; 
		// Execution time for single kernels
		

		//Part 3 - memory allocation
		// It doesn't matter which kernel we use here
		// It should set the size to as many items are in the work group.
		size_t local_size = kernel_reduceadd.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);

		size_t padding_size = A.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<mytype> A_ext(local_size-padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size()*sizeof(mytype);//size in bytes
		// How many elements in the vector compared to how many work items.
		size_t nr_groups = input_elements / local_size;

		//host - output
		// The vector for the mean.
		std::vector<mytype> B(input_elements);
		size_t output_size = B.size()*sizeof(mytype);//size in bytes
		// The vector which stores the minimum and maximum.
		std::vector<mytype> C(input_elements);

		// Print the vector for testing.
		std::cout << "Unsorted: " << A << std::endl;

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size); // The input
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size); // The mean
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_size); // min and max

		//Part 4 - device operations

		//4.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueWriteBuffer(buffer_C, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

		//4.2 Setup and execute all kernels (i.e. device code)
		// SORTING
		kernel_bitonicsort.setArg(0, buffer_A);
		// REDUCE ADD
		kernel_reduceadd.setArg(0, buffer_A);
		kernel_reduceadd.setArg(1, buffer_B);
		// MIN AND MAX
		kernel_minmax.setArg(0, buffer_A);
		kernel_minmax.setArg(1, buffer_C);

		//call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_reduceadd, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
		queue.enqueueNDRangeKernel(kernel_bitonicsort, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
		queue.enqueueNDRangeKernel(kernel_minmax, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

		// Profiling tasks
		queue.enqueueNDRangeKernel(kernel_reduceadd, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &total_profile);
		queue.enqueueNDRangeKernel(kernel_bitonicsort, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &total_profile);
		queue.enqueueNDRangeKernel(kernel_minmax, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &total_profile);

		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, output_size, &C[0]);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, output_size, &A[0]);

		//4.4 single kernel execution times
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0], NULL, &A_profile);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, input_size, &B[0], NULL, &B_profile);
		queue.enqueueWriteBuffer(buffer_C, CL_TRUE, 0, input_size, &A[0], NULL, &C_profile);

		float mean = B[0] / A.size();

		// Outputs
		std::cout << "Sorted: " << A << std::endl; // The sorted vector
		std::cout << "Mean: " << std::fixed << std::setprecision(2) << mean; // The mean/average
		std::cout << " Median: " << A[std::floor(A.size() / 2)]; // The median
		std::cout << " Upper quartile: " << A[std::floor(A.size() * 0.75)]; // The upper quartile
		std::cout << " Lower quartile: " << A[std::floor(A.size() * 0.25)]; // The lower quartile
		std::cout << " Minimum: " << C[1]; // The min
		std::cout << " Maximum: " << C[0] << std::endl; // The max

		// Reporting the kernel execution times
		std::cout << "Total execution time: " << total_profile.getProfilingInfo<CL_PROFILING_COMMAND_END>() - total_profile.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		std::cout << "  Buffer A execution time: " << A_profile.getProfilingInfo<CL_PROFILING_COMMAND_END>() - A_profile.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		std::cout << " Buffer B execution time: " << B_profile.getProfilingInfo<CL_PROFILING_COMMAND_END>() - B_profile.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		std::cout << " Buffer C execution time: " << C_profile.getProfilingInfo<CL_PROFILING_COMMAND_END>() - C_profile.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
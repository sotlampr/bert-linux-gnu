CPPFLAGS = -I./libtorch/include/ \
					-I./libtorch/include/torch/csrc/api/include

PREFIX := $(shell python -c "import sys; print(sys.exec_prefix)")
INCPATH := $(shell python -c "import torch.utils.cpp_extension as C; print('-I' + str.join(' -I', C.include_paths()))")
LIBPATH := $(shell python -c "import torch.utils.cpp_extension as C; print(C.include_paths()[0] + '/../lib')")
LIBS = -ltorch -lc10 -lc10_cuda -lcuda -lpthread
CXXFLAGS = -march=native -Ofast -pipe -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0

mnist: mnist.cpp
	if [ ! -e download_mnist.py ]; then wget "https://raw.githubusercontent.com/pytorch/pytorch/master/tools/download_mnist.py"; fi
	python download_mnist.py -ddata
	$(CXX) mnist.cpp -o mnist $(INCPATH) -L$(LIBPATH) -Wl,-rpath,$(LIBPATH) $(CXXFLAGS) $(LIBS) 

clean:
	rm -rf mnist download_mnist.py* data


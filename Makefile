PREFIX := $(shell python -c "import sys; print(sys.exec_prefix)")
INC := $(shell python -c "import torch.utils.cpp_extension as C; print('-I' + str.join(' -I', C.include_paths()))")
LIB := -L./lib
TORCHLIBS := $(shell python -c "import torch.utils.cpp_extension as C; print(C.include_paths()[0] + '/../lib')")
LIBS = -ltorch -lc10 -lc10_cuda -ltorch_cpu -lcuda -lpthread -licuuc -licuio
CXXFLAGS = -march=native -Og -pipe -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0


mnist: mnist.cpp
	if [ ! -e download_mnist.py ]; then wget "https://raw.githubusercontent.com/pytorch/pytorch/master/tools/download_mnist.py"; fi
	python download_mnist.py -ddata
	$(CXX) mnist.cpp -o mnist $(INC) $(LIB) -L$(TORCHLIBS) -Wl,-rpath,$(TORCHLIBS) $(CXXFLAGS) $(LIBS) 

tokenize: tokenize.cpp
	$(CXX) tokenize.cpp -o tokenize $(INC) $(LIB) -L$(TORCHLIBS) -Wl,-rpath,$(TORCHLIBS) $(CXXFLAGS) $(LIBS) 

state: lib/bert.o lib/state.o
	$(CXX) lib/state.o lib/bert.o -o state $(INC) $(LIB) -L$(TORCHLIBS) -Wl,-rpath,$(TORCHLIBS) $(CXXFLAGS) $(LIBS) 

lib/state.o: src/state.cpp
	$(CXX) -c src/state.cpp -o lib/state.o $(INC) $(LIB) -L$(TORCHLIBS) -Wl,-rpath,$(TORCHLIBS) $(CXXFLAGS) $(LIBS) 

lib/bert.o: src/bert.cpp
	$(CXX) -c src/bert.cpp -o lib/bert.o $(INC) $(LIB) -L$(TORCHLIBS) -Wl,-rpath,$(TORCHLIBS) $(CXXFLAGS) $(LIBS) 

clean:
	rm -rf mnist download_mnist.py* data tokenize

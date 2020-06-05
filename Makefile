PREFIX := $(shell python -c "import sys; print(sys.exec_prefix)")
INCLUDE := $(shell python -c "import torch.utils.cpp_extension as C; print('-I' + str.join(' -I', C.include_paths()))")
TORCHLIBS := $(shell python -c "import torch.utils.cpp_extension as C; print(C.include_paths()[0] + '/../lib')")
LDFLAGS := -ltorch -lc10 -lc10_cuda -ltorch_cpu -lcuda -lpthread -licuuc -licuio
CXXFLAGS := -march=native -Og -pipe -std=c++14 -ggdb3
CPPFLAGS := -D_GLIBCXX_USE_CXX11_ABI=0
BINDIR := bin
SRC_DIR := src
OBJ_DIR := build
SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES))

glue: mkgluedir download_glue_script download_glue
download_glue_script: glue/download_glue_data.py
download_glue: glue/data
process_cola: glue/data/CoLA/processed

mkgluedir:
	mkdir -p glue

glue/download_glue_data.py: mkgluedir
	wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py \
  -O glue/download_glue_data.py

glue/data: mkgluedir glue/download_glue_data.py
	python glue/download_glue_data.py --data_dir=glue/data --tasks=all

glue/data/CoLA/processed:
	mkdir -p glue/data/CoLA/processed
	for f in glue/data/CoLA/*.tsv; do \
	  cut -f2 -d'	' $$f > glue/data/CoLA/processed/$$(basename $$f .tsv)-labels; \
	  cut -f4 -d'	' $$f > glue/data/CoLA/processed/$$(basename $$f .tsv)-texts; \
	done

mnist: mnist.cpp
	if [ ! -e download_mnist.py ]; then wget "https://raw.githubusercontent.com/pytorch/pytorch/master/tools/download_mnist.py"; fi
	python download_mnist.py -ddata
	$(CXX) mnist.cpp -o mnist $(INC) $(LIB) -L$(TORCHLIBS) -Wl,-rpath,$(TORCHLIBS) $(CXXFLAGS) $(LIBS) 

tokenize: tokenize.cpp
	$(CXX) tokenize.cpp -o tokenize $(INC) $(LIB) -L$(TORCHLIBS) -Wl,-rpath,$(TORCHLIBS) $(CXXFLAGS) $(LIBS) 

train: $(OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS) -L$(TORCHLIBS) -Wl,-rpath,$(TORCHLIBS) 

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX)  -c -o $@ $< $(INCLUDE) $(CPPFLAGS) $(CXXFLAGS)

clean:
	rm build/* train

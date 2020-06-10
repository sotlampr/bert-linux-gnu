PREFIX := $(shell python -c "import sys; print(sys.exec_prefix)")
INCLUDE := $(shell python -c "import torch.utils.cpp_extension as C; print('-I' + str.join(' -I', C.include_paths()))")
TORCHLIBS := $(shell python -c "import torch.utils.cpp_extension as C; print(C.include_paths()[0] + '/../lib')")
LDFLAGS := -ltorch -lc10 -lc10_cuda -ltorch_cpu -lcuda -lpthread -licuuc -licuio
CXXFLAGS := -march=native -O0 -pipe -std=c++14 -ggdb3 -g
CPPFLAGS := -D_GLIBCXX_USE_CXX11_ABI=0 # -DDEBUG

MODULES := data metrics model state tokenize train
SRC_DIR := $(addprefix src/,$(MODULES))
BUILD_DIR := $(addprefix build/,$(MODULES))
SOURCES := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
OBJECTS := $(patsubst src/%.cpp,build/%.o,$(SOURCES))
INCLUDE += -I./src

vpath %.cpp $(SRC_DIR)

define make-goal
$1/%.o: %.cpp
	$(CXX)  -c $$< -o $$@ $(INCLUDE) $(CPPFLAGS) $(CXXFLAGS)
endef

.PHONY: all checkdirs clean

all: checkdirs train

bert: src/bert.cpp $(OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS) -L$(TORCHLIBS) -Wl,-rpath,$(TORCHLIBS) 
	
checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $@

clean:
	@rm -rf $(BUILD_DIR)
	@rm bert

$(foreach bdir,$(BUILD_DIR),$(eval $(call make-goal,$(bdir))))


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
	  cut -f2 -d'	' $$f | sed 's/\.0//g' > glue/data/CoLA/processed/$$(basename $$f .tsv)-labels; \
	  cut -f4 -d'	' $$f | sed 's/\.0//g' > glue/data/CoLA/processed/$$(basename $$f .tsv)-texts; \
	done

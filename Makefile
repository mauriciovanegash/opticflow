#****************************************************************************

# DEBUG can be set to YES to include debugging info, or NO otherwise
DEBUG          := NO

# PROFILE can be set to YES to include profiling info, or NO otherwise
PROFILE        := NO

#****************************************************************************
DEBUG_CXXFLAGS   := -Wall -Wno-format -g -DDEBUG
RELEASE_CXXFLAGS := -Wall -Wno-unknown-pragmas -Wno-format -O3

DEBUG_LDFLAGS    := -g
RELEASE_LDFLAGS  :=

ifeq (YES, ${DEBUG})
   CXXFLAGS     := ${DEBUG_CXXFLAGS}
   LDFLAGS      := ${DEBUG_LDFLAGS}
   TYPE 		= Debug
else
   CXXFLAGS     := ${RELEASE_CXXFLAGS}
   LDFLAGS      := ${RELEASE_LDFLAGS}
   TYPE 		= Release
endif

ifeq (YES, ${PROFILE})
   CXXFLAGS := ${CXXFLAGS} -pg -O3
   LDFLAGS  := ${LDFLAGS} -pg
endif
#****************************************************************************
# Preprocessor directives
#****************************************************************************
DEFS =
#DEFS += -DSHOWINGWARNINGS
#DEFS += -DSHOWINGINFO
#DEFS += -DRAPIDXML_NO_EXCEPTIONS
#DEFS += -DSAVE_ROTVEL_PROFILE
#DEFS += -DSAVE_TRANSVEL_PROFILE

CUDEFS =
CUDEFS += -DCUDA_TOOLKIT_10

ARCH := $(shell uname -m)
CXX = g++
CC = gcc

CXXFLAGS :=	${CXXFLAGS} ${DEFS} $(shell pkg-config --cflags opencv)

LIBS = -pthread $(shell pkg-config --libs opencv)

#****************************************************************************
# CUDA GPU directives
#****************************************************************************
CUDA_INSTALL_PATH = /usr/local/cuda
NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
CUDA_INCL := -I"$(CUDA_INSTALL_PATH)/include"

ifeq "$(ARCH)" "x86_64"
CUDALIBS += -L"$(CUDA_INSTALL_PATH)/lib64" -lcuda -lcudart -lcublas -lcufft
else
CUDALIBS += -L"$(CUDA_INSTALL_PATH)/lib" -lcuda -lcudart -lcublas -lcufft
endif

NVCCFLAGS := --compiler-options -fPIC --gpu-architecture=compute_53 --gpu-code=sm_53 $(CUDEFS) --ptxas-options=-v -O3 -G -g

#****************************************************************************
# Directories
#****************************************************************************
DIRECTORIES = src

# Add directories to the include and library paths
INCPATH = $(DIRECTORIES)
LIBPATH =

# Which files to add to backups, apart from the source code
EXTRA_FILES = Makefile

# Where to store object files.
STORE = $(TYPE)
# Makes a list of the source (.cpp) files.
CXXSOURCE := $(foreach DIR,$(DIRECTORIES),$(wildcard $(DIR)/*.cpp))
CSOURCE   := $(foreach DIR,$(DIRECTORIES),$(wildcard $(DIR)/*.c))
CUSOURCE  := $(foreach DIR,$(DIRECTORIES),$(wildcard $(DIR)/*.cu))
# List of header files.
HEADERS := $(foreach DIR,$(DIRECTORIES),$(wildcard $(DIR)/*.h))
HEADERS += $(foreach DIR,$(DIRECTORIES),$(wildcard $(DIR)/*.hpp))
HEADERS += $(foreach DIR,$(DIRECTORIES),$(wildcard $(DIR)/*.cuh))
# Makes a list of the object files that will have to be created.
OBJECTS  := $(addprefix $(STORE)/, $(CUSOURCE:.cu=.cu_o))
OBJECTS  += $(addprefix $(STORE)/, $(CSOURCE:.c=.o))
OBJECTS  += $(addprefix $(STORE)/, $(CXXSOURCE:.cpp=.o))
DFILES   := $(addprefix $(STORE)/,$(CXXFILES:.cpp=.d))
DFILES   += $(addprefix $(STORE)/,$(CFILES:.c=.d))

TARGET =	opticflow

# Specify phony rules. These are rules that are not real files.
.PHONY: clean backup dirs

$(TARGET): dirs $(OBJECTS)
		@echo 'Building target: $@'
		@echo 'Invoking: GCC C++ Linker'
		$(CXX) -o $(TARGET) $(OBJECTS) $(LIBS) $(CUDALIBS)
		@echo 'Finished building target: $@'
		@echo ' '

$(STORE)/%.o: %.cpp
	@echo 'Building partial codes: $^'
	$(CXX) -Wp,-MMD,$(STORE)/$*.dd $(CXXFLAGS) $(foreach INC,$(INCPATH),-I$(INC)) -c $^ -o $@
	@sed -e '1s/^\(.*\)$$/$(subst /,\/,$(dir $@))\1/' $(STORE)/$*.dd > $(STORE)/$*.d
	@echo ' '

$(STORE)/%.o: %.c
	@echo 'Building partial codes: $^'
	$(CXX) -Wp,-MMD,$(STORE)/$*.dd $(CXXFLAGS) $(foreach INC,$(INCPATH),-I$(INC)) -c $^ -o $@
	@sed -e '1s/^\(.*\)$$/$(subst /,\/,$(dir $@))\1/' $(STORE)/$*.dd > $(STORE)/$*.d
	@echo ' '

$(STORE)/%.cu_o: %.cu
	@echo 'Building partial codes: $^'
	$(NVCC) $(NVCCFLAGS) $(CUDA_INCL) -c $^ -o $@
	@echo ' '

# Empty rule to prevent problems when a header is deleted.
%.hpp: ;

all:	$(TARGET)

# Cleans up the objects, .d files and executables.
clean:
	@echo 'Cleaning all!!!'
	@-rm -f $(foreach DIR,$(DIRECTORIES),$(STORE)/$(DIR)/*.d $(STORE)/$(DIR)/*.dd $(STORE)/$(DIR)/*.o $(STORE)/$(DIR)/*.cu_o)
	@-rm -f $(TARGET)
	@echo ' '

# Create necessary directories
dirs:
	@-if [ ! -e $(STORE) ]; then mkdir $(STORE); fi;
	@-$(foreach DIR,$(DIRECTORIES), if [ ! -e $(STORE)/$(DIR) ]; then mkdir $(STORE)/$(DIR); fi; )

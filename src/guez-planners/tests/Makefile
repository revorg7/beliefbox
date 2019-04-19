SMPL_DIR := $(shell cd ../..; pwd)

include $(SMPL_DIR)/Make-default.mk

%: %.cc
	cd $(SMPL_DIR); ${MAKE}
	mkdir -p bin/
	$(CXX) $(CFLAGS) -shared $(INCS) -DMAKE_MAIN -o bin/$@.so $< ${MAIN_LIB} $(LIBS) `python3-config --ldflags`


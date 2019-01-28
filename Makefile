all:
	python setup.py build_ext --inplace

clean:
	python setup.py clean --all
#	rm -f LakeCC.so cython/LakeCC.cpp

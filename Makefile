CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

all: figures/figure1.pdf \
		figures/figure3.pdf \
		figures/figure4a.pdf \
		figures/figure4b.pdf \
		figures/figure5a.pdf \
		figures/figure5b.pdf \
		figures/figure6.pdf

figures/figure%.pdf: python/figure%.py
	$(CONDA_ACTIVATE) quantum_spin_dynamics && python $< >> $(@:.pdf=.log) 2>&1

clean:
	-rm -rf figures/*
	-rm -rf python/__pycache__

.PHONY: all clean
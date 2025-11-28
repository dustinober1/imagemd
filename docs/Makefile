# Makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = source
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets
clean:
	rm -rf $(BUILDDIR)/*

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml:
	@$(SPHINXBUILD) -b dirhtml "$(SOURCEDIR)" "$(BUILDDIR)/dirhtml" $(SPHINXOPTS) $(O)
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

singlehtml:
	@$(SPHINXBUILD) -b singlehtml "$(SOURCEDIR)" "$(BUILDDIR)/singlehtml" $(SPHINXOPTS) $(O)
	@echo "Build finished. The HTML page is in $(BUILDDIR)/singlehtml."

pickle:
	@$(SPHINXBUILD) -b pickle "$(SOURCEDIR)" "$(BUILDDIR)/pickle" $(SPHINXOPTS) $(O)
	@echo "Build finished; now you can process the pickle files."

json:
	@$(SPHINXBUILD) -b json "$(SOURCEDIR)" "$(BUILDDIR)/json" $(SPHINXOPTS) $(O)
	@echo "Build finished; now you can process the JSON files."

htmlhelp:
	@$(SPHINXBUILD) -b htmlhelp "$(SOURCEDIR)" "$(BUILDDIR)/htmlhelp" $(SPHINXOPTS) $(O)
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp:
	@$(SPHINXBUILD) -b qthelp "$(SOURCEDIR)" "$(BUILDDIR)/qthelp" $(SPHINXOPTS) $(O)
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(BUILDDIR)/qthelp, like this:"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/VisionPDF.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/VisionPDF.qhc"

devhelp:
	@$(SPHINXBUILD) -b devhelp "$(SOURCEDIR)" "$(BUILDDIR)/devhelp" $(SPHINXOPTS) $(O)
	@echo "Build finished."
	@echo "To view the help file:"
	@echo "# assistant -help $(BUILDDIR)/devhelp/VisionPDF"

epub:
	@$(SPHINXBUILD) -b epub "$(SOURCEDIR)" "$(BUILDDIR)/epub" $(SPHINXOPTS) $(O)
	@echo "Build finished. The epub file is in $(BUILDDIR)/epub."

epub3:
	@$(SPHINXBUILD) -b epub3 "$(SOURCEDIR)" "$(BUILDDIR)/epub3" $(SPHINXOPTS) $(O)
	@echo "Build finished. The epub3 file is in $(BUILDDIR)/epub3."

latex:
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS) $(O)
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."

latexpdf:
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS) $(O)
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C "$(BUILDDIR)/latex" all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

latexpdfja:
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS) $(O)
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C "$(BUILDDIR)/latex" all-pdf-ja
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

text:
	@$(SPHINXBUILD) -b text "$(SOURCEDIR)" "$(BUILDDIR)/text" $(SPHINXOPTS) $(O)
	@echo "Build finished. The text files are in $(BUILDDIR)/text."

man:
	@$(SPHINXBUILD) -b man "$(SOURCEDIR)" "$(BUILDDIR)/man" $(SPHINXOPTS) $(O)
	@echo "Build finished. The manual pages are in $(BUILDDIR)/man."

texinfo:
	@$(SPHINXBUILD) -b texinfo "$(SOURCEDIR)" "$(BUILDDIR)/texinfo" $(SPHINXOPTS) $(O)
	@echo "Build finished. The Texinfo files are in $(BUILDDIR)/texinfo."

info:
	@$(SPHINXBUILD) -b texinfo "$(SOURCEDIR)" "$(BUILDDIR)/texinfo" $(SPHINXOPTS) $(O)
	@echo "Running Texinfo files through makeinfo..."
	make -C "$(BUILDDIR)/texinfo" info
	@echo "makeinfo finished; the info files are in $(BUILDDIR)/texinfo."

gettext:
	@$(SPHINXBUILD) -b gettext "$(SOURCEDIR)" "$(BUILDDIR)/locale" $(SPHINXOPTS) $(O)
	@echo "Build finished. The gettext files are in $(BUILDDIR)/locale."

changes:
	@$(SPHINXBUILD) -b changes "$(SOURCEDIR)" "$(BUILDDIR)/changes" $(SPHINXOPTS) $(O)
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
	@echo "Link check complete; look for any errors in the above output" \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	@$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS) $(O)
	@echo "Testing of doctests in the sources finished, look at the" \
	      "results in $(BUILDDIR)/doctest/output.txt."

coverage:
	@$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(SPHINXOPTS) $(O)
	@echo "Testing of coverage in the sources finished, look at the" \
	      "results in $(BUILDDIR)/coverage/python.txt."

xml:
	@$(SPHINXBUILD) -b xml "$(SOURCEDIR)" "$(BUILDDIR)/xml" $(SPHINXOPTS) $(O)
	@echo "Build finished. The XML files are in $(BUILDDIR)/xml."

pseudoxml:
	@$(SPHINXBUILD) -b pseudoxml "$(SOURCEDIR)" "$(BUILDDIR)/pseudoxml" $(SPHINXOPTS) $(O)
	@echo "Build finished. The pseudo-XML files are in $(BUILDDIR)/pseudoxml."

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Development targets
dev-setup:
	pip install -e .[docs]
	pip install sphinx-autobuild

serve:
	cd _build/html && python -m http.server 8000
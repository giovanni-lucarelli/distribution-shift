### Usage

Put the template folder and the project folder in the same directory.

#### Inside your project

Inside your project use the following structure:
- `chapters` folder contains all chapters, including abstract and conclusion.
- `main.tex` is the main file, there you can define title page settings (as university, faculty, ...) and include all the chapters using the `\includechapters` macro.
- `references.bib` shoul include bibitems to cite using `\cite{<label>}` in .tex files.

> [!WARNING]
> - use `chapters` as name for the chapter folder and `ch<n>.tex` (ch1.tex, ch2.tex, ...) to name the single chapters inside, or the macro `\includechapters` won't work.
> - You can put any file you want in `chapter` folder, but then you'll have to add them manually (as abstract and conclusion)

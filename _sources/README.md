# Socialdata22_projectB
Project B website


# To build local website
- cd out of Social.../
- conda activate env with jupyter-book
- jb build Social.../
- open a .html file

# Host website 
- pip install ghp-import
- From the main branch of your book’s root directory (which should contain the _build/html folder) call ghp-import and point it to your HTML files, like so: ghp-import -n -p -f _build/html
- Now the website is live. Go to repo on GitHub -> Settings -> Pages   to find the link of your website

#!/usr/bin/env just --justfile

default: serve
  # @just --list

time := `date "+%Y-%m-%d %H:%M:%S"`

install:
    brew install hugo

serve:
    hugo server

publish:
    hugo

push: publish
    #!/bin/sh
    echo "current time is: {{time}}"
    git config user.name desonglll
    git config user.email lindesong66@163.com
    git add --ignore-removal .
    git commit -m "last updated at {{time}} by `git config user.name`"
    git push -u origin main

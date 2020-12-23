## if we are processing a method function
%if header['function']:
    %if header['class']:
${h3} .${header['function']}
    %else:
${h3} ${header['function']}
    %endif
%if not source is UNDEFINED:
<a href="${source}">
<img height="16px" width="16px" src="/img/GitHub-Mark-32px.png">
source
</a>
%endif
```python
.${signature}
```
%elif header['class']:
${h2} ${header['class']}
%if not source is UNDEFINED:
<a href="${source}">
<img height="16px" width="16px" src="/img/GitHub-Mark-32px.png">
source
</a>
%endif
```python 
${signature}
```

%endif 

%for section in sections:
    %if section['header']:

**${section['header']}**

    %else:
---
    %endif
    %if section['args']:
        %for arg in section['args']:
        %if arg['field']:
* **${arg['field']}** ${arg['signature']} : ${arg['description']}
        %else:
* ${arg['description']}
        %endif
        %endfor
    %endif
${section['text']}
%endfor



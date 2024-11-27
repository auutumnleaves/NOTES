---
year: {{date | format("YYYY")}}
tags: {% for t in tags %}{{t.tag}}{% if not loop.last %}, {% endif %}{% endfor %}
authors: {{authors}}{{directors}}
--- 

---
# {{title}}
{% for annotation in annotations -%}
  {% if annotation.comment %}
- {{annotation.comment}} [link](zotero://open-pdf/library/items/{{annotation.attachment.itemKey}}?page={{annotation.page}}&annotation={{annotation.id}})
  {% endif %}
  {%- if annotation.annotatedText -%}
- [{{annotation.annotatedText}} ](zotero://open-pdf/library/items/{{annotation.attachment.itemKey}}?page={{annotation.page}}&annotation={{annotation.id}})
  {%- endif %}
  {%- if annotation.imageRelativePath -%}
  ![[{{annotation.imageRelativePath}}|400]]
  {%- endif %}
{% endfor -%}
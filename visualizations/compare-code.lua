--[[
compare-code.lua: the PDF version of the widgets' "Compare code" view.

A widget (loop-widgets.js, stan-scatter-widget.js) is a div that holds a
```stan and a ```python code block. In HTML, the widget shows them. In the
PDF, which has no widget, this filter sets them next to each other under the
titles "Stan" and "JAX", as the widget's code view does: two equal columns
when each block fits in half the width, else the wider block at its own width
and the other in the rest, or one above the other when the two do not fit
side by side. The step-by-step views have no PDF
version. To keep such a figure where it is in the text rather than floating,
give its div fig-pos="H" in the .qmd.
]]

if not quarto.doc.is_format("latex") then
  return {}
end

local HEADER = [[
\newlength{\ccChar}\newlength{\ccFree}\newlength{\ccHalf}\newlength{\ccStan}\newlength{\ccJax}
\newcommand{\ccGap}{1.5em}
% \ccWidths{<longest Stan line>}{<longest JAX line>}, in characters, sets the
% widths of the two code columns: side by side if they fit, else stacked.
% Side by side, both are half the width unless one block is wider than that.
\newcommand{\ccWidths}[2]{%
  \settowidth{\ccChar}{\ttfamily 0}%
  \setlength{\ccFree}{\dimexpr\linewidth-#1\ccChar-#2\ccChar-\ccGap\relax}%
  \ifdim\ccFree<0pt
    \setlength{\ccStan}{\linewidth}\setlength{\ccJax}{\linewidth}%
    \def\ccBetween{\par\medskip\noindent}%
  \else
    \setlength{\ccHalf}{\dimexpr(\linewidth-\ccGap)/2\relax}%
    \setlength{\ccStan}{\ccHalf}%
    \ifdim#1\ccChar>\ccHalf \setlength{\ccStan}{#1\ccChar}\fi
    \ifdim#2\ccChar>\ccHalf \setlength{\ccStan}{\dimexpr\linewidth-\ccGap-#2\ccChar\relax}\fi
    \setlength{\ccJax}{\dimexpr\linewidth-\ccStan-\ccGap\relax}% the rest, so rounding never overfills the line
    \def\ccBetween{\hfill}%
  \fi}
\newcommand{\ccTitle}[1]{{\raggedright\sffamily\bfseries\small\color[HTML]{5F5E5A}#1\par}}
]]

local header_added = false

local function is_widget(block)
  return block.t == "Div" and (block.classes:includes("loop-widget") or block.classes:includes("stan-scatter"))
end

local function longest(code)
  local n = 0
  for line in (code.text .. "\n"):gmatch("(.-)\n") do
    n = math.max(n, utf8.len(line) or #line)
  end
  return n
end

-- The widget becomes its two code blocks, in two columns.
local function compare(div)
  if not is_widget(div) then return nil end
  local stan, jax
  for _, b in ipairs(div.content) do
    if b.t == "CodeBlock" and b.classes:includes("stan") then stan = b end
    if b.t == "CodeBlock" and b.classes:includes("python") then jax = b end
  end
  if not (stan and jax) then return nil end
  if not header_added then
    quarto.doc.include_text("in-header", HEADER)
    header_added = true
  end
  local raw = function(s) return pandoc.RawBlock("latex", s) end
  return {
    raw(string.format("\\ccWidths{%d}{%d}\\par\\noindent\n\\begin{minipage}[t]{\\ccStan}\n\\ccTitle{Stan}",
                      longest(stan), longest(jax))),
    stan,
    raw("\\end{minipage}%\n\\ccBetween\n\\begin{minipage}[t]{\\ccJax}\n\\ccTitle{JAX}"),
    jax,
    raw("\\end{minipage}\\par"),
  }
end

return {
  { Div = compare },
}

#import "@preview/cetz:0.5.3"
#cetz.canvas(length: 0.75pt, {
import cetz.draw: *
rect((0, 0), (571, -670), stroke: none, fill: none)
rect((0, 0), (571, -670), fill: rgb(255, 255, 255), stroke: none)
line((28.7, -558), (27.7, -558), (27.7, -70.4), (28.7, -70.4), close: true, fill: rgb(0, 114, 178), stroke: (paint: rgb(128, 128, 128), thickness: 0.6pt, join: "round", cap: "round"))
line((28.7, -584), (27.7, -584), (27.7, -564.2), (28.7, -564.2), close: true, fill: rgb(224, 224, 224), stroke: (paint: rgb(128, 128, 128), thickness: 0.6pt, join: "round", cap: "round"))
line((33.9, -90.2), (542.3, -90.2), (542.3, -70.4), (33.9, -70.4), close: true, fill: rgb(204, 227, 240), stroke: (paint: rgb(128, 128, 128), thickness: 0.6pt, join: "round", cap: "round"))
line((33.9, -584), (542.3, -584), (542.3, -96.4), (33.9, -96.4), close: true, fill: rgb(224, 224, 224), stroke: (paint: rgb(128, 128, 128), thickness: 0.6pt, join: "round", cap: "round"))
content((288.1, -330.8), anchor: "base", text(size: 15.594pt, "Diseaseᶜ ∩ Positiveᶜ"))
content((288.1, -344.4), anchor: "north", text(size: 15.594pt, "0.94"))
content((28.2, -49.1), anchor: "center", text(size: 17.544pt, "Disease"))
content((288.1, -49.1), anchor: "center", text(size: 17.544pt, "Diseaseᶜ"))
content((285.5, -613.1), anchor: "center", text(size: 16.374pt, "P(Disease | Positive) = P(Disease ∩ Positive) / P(Positive) = 0.0095 / 0.059 = 0.161"))
})

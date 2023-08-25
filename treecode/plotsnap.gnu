set terminal gif
set xrange [-20:20]
set yrange [-20:20]
TITLE="T=`head -1 data | awk '{print $1}'`"
plot "data" using 2:3 every ::1::10000 title TITLE

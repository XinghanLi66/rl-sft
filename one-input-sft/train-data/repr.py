multi = r"""To determine the wind velocity when the pressure on a sail of 4 square feet is 32 pounds, we start by expressing the relationship between pressure \( P \), area \( A \), and velocity \( V \) mathematically. Since the pressure varies jointly as the area of the sail and the cube of the wind's velocity, we can write:

\[ P = kA V^3 \]

where \( k \) is a constant of proportionality. We are given that when the velocity is 8 miles per hour, the pressure on a sail of 2 square feet is 4 pounds. We can use this information to find the value of \( k \). Substituting \( P = 4 \), \( A = 2 \), and \( V = 8 \) into the equation, we get:

\[ 4 = k \cdot 2 \cdot 8^3 \]

First, we calculate \( 8^3 \):

\[ 8^3 = 512 \]

So the equation becomes:

\[ 4 = k \cdot 2 \cdot 512 \]

Next, we multiply 2 by 512:

\[ 2 \cdot 512 = 1024 \]

Thus, the equation simplifies to:

\[ 4 = k \cdot 1024 \]

To solve for \( k \), we divide both sides of the equation by 1024:

\[ k = \frac{4}{1024} = \frac{1}{256} \]

Now that we have the value of \( k \), we can use it to find the wind velocity when the pressure on a sail of 4 square feet is 32 pounds. Substituting \( P = 32 \), \( A = 4 \), and \( k = \frac{1}{256} \) into the equation \( P = kA V^3 \), we get:

\[ 32 = \frac{1}{256} \cdot 4 \cdot V^3 \]

First, we simplify the right side of the equation:

\[ \frac{1}{256} \cdot 4 = \frac{4}{256} = \frac{1}{64} \]

So the equation becomes:

\[ 32 = \frac{1}{64} V^3 \]

To solve for \( V^3 \), we multiply both sides of the equation by 64:

\[ V^3 = 32 \cdot 64 \]

Next, we calculate \( 32 \cdot 64 \):

\[ 32 \cdot 64 = 2048 \]

Thus, the equation simplifies to:

\[ V^3 = 2048 \]

To find \( V \), we take the cube root of both sides of the equation:

\[ V = \sqrt[3]{2048} = 12 \]

Therefore, the wind velocity when the pressure on 4 square feet of sail is 32 pounds is \(\boxed{12}\) miles per hour.
"""

single = repr(multi)

with open("repr.out", "w") as f:
    f.write(single)
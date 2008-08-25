## -*- Mode: octave -*-

n_arms=2;

iter = 14;

X=load ("results/belief_uct/bandit/baseline_long_double.out");
Y=load ("results/belief_uct/bandit/uct_complete.out");

hold off
s=X(:,1)==0 & X(:,3)==n_arms;
plot(1./(1-X(s,2)), X(s,6) - X(s,4), '0@-;ucb;');
hold on
s=X(:,1)==3 & X(:,3)==n_arms;
plot(1./(1-X(s,2)), X(s,6) - X(s,4), '1-;biased beta;');

##s=X(:,1)==4 & X(:,3)==n_arms;
##plot(1./(1-X(s,2)), X(s,6) - X(s,4), '2-;greedy;');

s=Y(:,1)==0 & Y(:,2)==iter & Y(:,4)==n_arms;
plot(1./(1-Y(s,3)), Y(s,7) - Y(s,5), '2@-;serial;');


s=Y(:,1)==1 & Y(:,2)==iter & Y(:,4)==n_arms;
plot(1./(1-Y(s,3)), Y(s,7) - Y(s,5), '3@-;random;');

s=Y(:,1)==3 & Y(:,2)==iter & Y(:,4)==n_arms;
loglog(1./(1-Y(s,3)), Y(s,7) - Y(s,5), '4@-;LB;');

s=Y(:,1)==9 & Y(:,2)==iter & Y(:,4)==n_arms;
loglog(1./(1-Y(s,3)), Y(s,7) - Y(s,5), '5@-;UB;');



grid on;
#title(strcat("n_arms = ",  num2str(n_arms));
print("bandit_baseline.eps");

const colors = {
  easy: "bg-green-100 text-green-800",
  medium: "bg-yellow-100 text-yellow-800",
  hard: "bg-red-100 text-red-800",
  default: "bg-gray-100 text-gray-800",
};

interface BadgeProps {
  variant?: keyof typeof colors;
  children: React.ReactNode;
}

export default function Badge({ variant = "default", children }: BadgeProps) {
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colors[variant]}`}>
      {children}
    </span>
  );
}

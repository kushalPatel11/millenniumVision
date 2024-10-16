// src/components/SmoothScrollHero.tsx
import { ReactLenis } from "lenis/dist/lenis-react";
import {
  motion,
  useMotionTemplate,
  useScroll,
  useTransform,
} from "framer-motion";
import { SiSpacex } from "react-icons/si";
import { FiMapPin } from "react-icons/fi";
import { useRef } from "react";

// Define props interface
interface SmoothScrollHeroProps {
  onLogin: () => void; // Define the onLogin prop
}

export const SmoothScrollHero = ({ onLogin }: SmoothScrollHeroProps) => {
  return (
    <div className="bg-zinc-950">
      <ReactLenis
        root
        options={{
          lerp: 0.05,
        }}
      >
        <Nav onLogin={onLogin} />  {/* Pass onLogin to Nav */}
        <Hero />
        <Schedule />
      </ReactLenis>
    </div>
  );
};

const Nav = ({ onLogin }: { onLogin: () => void }) => {  // Accept onLogin in Nav props
  return (
    <nav className="fixed left-0 right-0 top-0 z-50 flex items-center justify-between px-6 py-3 text-white">
      <button> </button>
      <button
        onClick={onLogin} // Call onLogin to navigate to VideoUpload
        className="flex items-center gap-1 text-s text-black bg-yellow-400 hover:bg-yellow-500 rounded-full px-4 py-3 shadow-lg transition-all duration-300"
        style={{ fontFamily: 'PS2' }}
      >
        Analyze Gameplay
      </button>

    </nav>
  );
};

const SECTION_HEIGHT = 1500;

const Hero = () => {
  return (
    <div
      style={{ height: `calc(${SECTION_HEIGHT}px + 100vh)` }}
      className="relative w-full"
    >
      <CenterImage />
      <ParallaxImages />
      <div className="absolute bottom-0 left-0 right-0 h-96 bg-gradient-to-b from-zinc-950/0 to-zinc-950" />
    </div>
  );
};

const CenterImage = () => {
  const { scrollY } = useScroll();
  const clip1 = useTransform(scrollY, [0, 1500], [25, 0]);
  const clip2 = useTransform(scrollY, [0, 1500], [75, 100]);
  const clipPath = useMotionTemplate`polygon(${clip1}% ${clip1}%, ${clip2}% ${clip1}%, ${clip2}% ${clip2}%, ${clip1}% ${clip2}%)`;
  const backgroundSize = useTransform(
    scrollY,
    [0, SECTION_HEIGHT + 500],
    ["170%", "100%"]
  );
  const opacity = useTransform(
    scrollY,
    [SECTION_HEIGHT, SECTION_HEIGHT + 500],
    [1, 0]
  );

  return (
    <motion.div
      className="sticky top-0 h-screen w-full"
      style={{
        clipPath,
        backgroundSize,
        opacity,
        backgroundImage:
          "url(Img2.webp)",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
      }}
    />
  );
};

const ParallaxImages = () => {
  return (
    <div className="mx-auto max-w-5xl px-4 pt-[200px]">
      <ParallaxImg
        src="/Img3.webp"
        alt="Image"
        start={-200}
        end={200}
        className="w-1/3"
      />
      <ParallaxImg
        src="/Img5.webp"
        alt="An example of a space launch"
        start={200}
        end={-250}
        className="mx-auto w-2/3"
      />
      <ParallaxImg
        src="/Img4.webp"
        alt="Orbiting satellite"
        start={-200}
        end={200}
        className="ml-auto w-1/3"
      />
      <ParallaxImg
        src="/Img1.webp"
        alt="Orbiting satellite"
        start={0}
        end={-500}
        className="ml-24 w-5/12"
      />
    </div>
  );
};

const ParallaxImg = ({ className, alt, src, start, end }: { className: any, alt: any, src: any, start: any, end: any }) => {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: [`${start}px end`, `end ${end * -1}px`],
  });

  const opacity = useTransform(scrollYProgress, [0.75, 1], [1, 0]);
  const scale = useTransform(scrollYProgress, [0.75, 1], [1, 0.85]);
  const y = useTransform(scrollYProgress, [0, 1], [start, end]);
  const transform = useMotionTemplate`translateY(${y}px) scale(${scale})`;

  return (
    <motion.img
      src={src}
      alt={alt}
      className={className}
      ref={ref}
      style={{ transform, opacity }}
    />
  );
};

const Schedule = () => {
  return (
    <section
      id="launch-schedule"
      className="mx-auto max-w-5xl px-4 py-48 text-white"
    >
      <motion.h1
        initial={{ y: 48, opacity: 0 }}
        whileInView={{ y: 0, opacity: 1 }}
        transition={{ ease: "easeInOut", duration: 0.75 }}
        className="mb-20 text-4xl font-black uppercase text-zinc-50"
      >
        Millennium Vision
      </motion.h1>
      <ScheduleItem title="SHOTS FIRED" count="29" />
      <ScheduleItem title="SHOTS HIT"  count="20" />
      <ScheduleItem title="HEADSHOTS"  count="7" />
      <ScheduleItem title="SHOTS MISSED"  count="9" />
      <ScheduleItem title="AIM ACCURACY"  count="6.8" />
      <ScheduleItem title="RECOILE CONTROL "  count="8" />
      <ScheduleItem title="OVERALL GAMEPLAY"  count="8" />
    </section>
  );
};

const ScheduleItem = ({ title, count }: { title: any, count: any }) => {
  return (
    <motion.div
      initial={{ y: 48, opacity: 0 }}
      whileInView={{ y: 0, opacity: 1 }}
      transition={{ ease: "easeInOut", duration: 0.75 }}
      className="mb-9 flex items-center justify-between border-b border-zinc-800 px-3 pb-9"
    >
      <div>
        <p className="mb-1.5 text-xl text-zinc-50">{title}</p>
      </div>
      <div className="mb-1.5 text-xl text-zinc-50">
        <p className="text-sm">{count}</p>
      </div>
    </motion.div>
  );
};
